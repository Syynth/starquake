# GPU front/back protocol and engine state model

Status: draft for the owner's review (issue #12). Not approved; nothing is built on it yet.

This spec designs `crates/gpu` from a GPFIFO submission to a recorded draw: how command buffers
are parsed, how a channel waits on a semaphore without stalling anything else, where macros run,
what crosses from the front thread to the back thread, how each engine keeps its state, how the
GPU reads and writes guest memory, and how a frame's command stream is captured and replayed
with no game and no GPU. It also settles where block-linear tiling lives (issue #17).

It builds on the approved execution model, kernel, services, observability and test strategy
specs, and on the existing Turing MME64 decoder and interpreter in `crates/gpu/src/mme`. Three
sibling specs are drafted at the same time: the graphics server session model (#11), the shader
pipeline (#13) and the render backend (#14). The section "Interfaces to neighbouring designs"
states exactly what this spec assumes of each and what it offers them.

## Summary

- **Two threads, one ring.** One front thread serves every channel: it reads command buffers,
  runs host methods, suspends a channel on an unmet semaphore acquire, runs macros, and drops
  register writes that change nothing. It sends what is left through one bounded
  single-producer, single-consumer ring to one back thread, which owns engine state, derives
  draws, dispatches and copies, and records them for the render backend. Eden-NX2 measured this
  split at 1.46x the draw throughput of one thread (rows 142–145).
- **The front owns the register file.** Macros read registers, and same-value dropping compares
  against them, so the front keeps the only authoritative copy and sends resolved writes. The
  back keeps a mirror it never writes on its own, plus dirty flags.
- **Only the waiting channel waits.** An unmet acquire parks that channel at the exact word
  (row 50). Other channels keep running, and the front never stalls on the back to check a
  semaphore: it rechecks when the back reports a release, when a submission arrives, or on a
  short poll for semaphores the CPU releases.
- **Releases happen after the work they follow.** A semaphore release or syncpoint increment
  is performed by the back, in ring order, once the host GPU has finished the recorded work
  before it (with Eden-NX2's empty-fence skip and batching, rows 123–124). Waiters a release
  satisfies are latched, so an equal-acquire never misses a value (row 70).
- **Ampere classes are modelled natively** (`C797` 3D, `C7C0` compute with QMD 3.0, `C7B5`
  DMA copy, `A140` inline-to-memory, `902D` 2D, the `C76F` host methods), from register tables
  generated from Mesa's class headers. Nothing is aliased to Maxwell.
- **Backend-neutral output.** The back hands the render backend plain-data commands (passes,
  draws, dispatches, copies, clears, fence actions) and asks the shader pipeline to translate a
  program read through a recorded environment. A recording backend makes the whole path
  testable without a GPU.
- **Capture and replay.** A capture holds the submissions, every byte each GPU thread read (in
  order, deduplicated), and the engine state at the start. Replaying it runs the real front and
  back on any host and diffs the ring and the recorded commands against goldens.
- **Block-linear tiling is a pure leaf crate** (`starquake-layout`), ported from Mesa's `nil`,
  safe Rust, tested offline. Eden-NX2's "Switch 2 GOB swizzle" (row 69) is Mesa's Turing colour
  GOB, by the derivation below.

## Sources and how facts are cited

- "Row N" is a row of Eden-NX2's blocker log, `eden-nx2/NX2_NOTES.md`, summarised in
  `docs/switch2-eden-diff-scoping.md`.
- Reference paths are relative to the reference folder (`tools/refs-path.sh`), at the commits
  pinned in `docs/references.md`: `eden-nx2` @ `50fa86bb`, `eden-pr4588` @ `e5fb3b97`, `eden` @
  `cf654052`, `mesa-nouveau` @ `52238762`. Mesa's class headers are under
  `mesa-nouveau/src/nouveau/headers/nvidia/classes/`, shortened below to `classes/`.
- Method numbers are dword indices (byte address / 4), as in `crates/gpu/src/mme`. Where a class
  header gives a byte address, both are written.
- A statement that no reference settles is marked **guess**, with what would confirm it.

### Protocol facts this design rests on

| Fact | Source |
|---|---|
| A GP entry is 8 bytes: word 0 bits 31:2 the low address, word 1 bits 7:0 the high address, bit 9 the level (main or subroutine), bits 30:10 the length in words, bit 31 sync | `classes/clc76f.h` (`NVC76F_GP_ENTRY0/1`); Eden's `CommandListHeader` in `eden-nx2/src/video_core/dma_pusher.h` agrees (address bits 0–39, size bits 42–62, sync bit 63) |
| A method header: dword address in bits 11:0, subchannel 15:13, count (or immediate data) 28:16, opcode 31:29: 1 incrementing, 3 non-incrementing, 4 immediate, 5 increment-once, 7 end of segment; 0 and 2 are subdevice-mask operations | `classes/clc76f.h` (`NVC76F_DMA_*`). Eden reads 13 address bits (`CommandHeader` in `dma_pusher.h`); we follow the header's 12 and pin the difference with a test |
| Methods below 0x40 are host methods: 0 bind object, 4–7 `SEMAPHOREA`–`D`, 8 non-stall interrupt, 0xA–0xD memory operations, 0x14 set reference, 0x17–0x1A semaphore address and payload, 0x1B `SEM_EXECUTE`, 0x1E wait for idle, 0x20 yield | `classes/clc76f.h`; `eden-nx2/src/video_core/engines/puller.cpp` |
| `SEMAPHORED` bits 4:0 the operation (1 acquire-equal, 2 release, 4 acquire-greater-or-equal, 8 acquire-and, 0x10 reduction), bit 24 release size (4 bytes when set), bits 30:27 the reduction | `classes/clc76f.h`; the 4-byte release is row 52 |
| Each command buffer is gated by a `SEMAPHORED` acquire the CPU releases after writing it; the GPU must stop at that word and resume there, reading the words after it again | Row 50; `eden-nx2/src/video_core/dma_pusher.cpp` (`command_resume_offset`) |
| A method's data may continue in the next GP entry of the channel, whatever that entry's address or level; parking split methods by resume address (row 56) was replaced by this | Row 73; `dma_pusher.cpp` @ `50fa86bb`, where parking is behind an opt-in variable |
| NVN2 binds `C797`, `C7C0`, `C7B5` and class 0 (unbind); it uses subchannel 2 as inline-to-memory without binding it | Rows 31, 49; `puller.cpp` (`ProcessBindMethod`) |
| 3D macros are called at methods 0xE00 and up (`CALL_MME_MACRO(j)` at byte 0x3800 + 8j); a call runs when its last data word arrives | `classes/clc797.h`; `eden-nx2/src/video_core/engines/maxwell_3d.cpp` (`ProcessMacro`) |
| The compute class also runs macros at 0xE00 and up, sharing its channel's data RAM with the 3D class; Mesa's `clc7c0.h` does not list these methods | `eden-nx2/src/video_core/engines/kepler_compute.cpp` (line 133), `control/channel_state.cpp` (`SetNativeDataRam`) |
| Macro data RAM is per channel | Row 84 |
| MME DMA reads: 3D methods 0x154/0x155 (`SET_MME_MEM_ADDRESS_A/B`, byte 0x550/0x554) and 0x158 (`MME_DMA_READ_FIFOED`, byte 0x560) feed the load FIFO after a load barrier | Row 58; `classes/clc797.h` |
| `SET_MME_SHADOW_RAM_CONTROL` is method 0x49 (byte 0x124), modes track, track with filter, passthrough, replay | `classes/clc797.h`; row 144 |
| About 60% of 3D register writes repeat the register's value | Row 135 |
| Compute dispatches use QMD 3.0 (word 18 bits 7:0 = 0x30), scheduled with `SEND_PCAS_A` (method 0xAD, QMD address >> 8) and `SEND_SIGNALING_PCAS2_B` (0xB0: action bits 3:0, count minus one bits 23:14) | `classes/clc7c0.h`, `classes/clc7c0qmd.h`; `eden-nx2/src/video_core/native_qmd.h` |
| QMD 3.0 release slots are at words 24, 28 and 52 (`RELEASE0/1/2`); enable is bit 23 of the slot's second word, reduction enable bit 26, reduction op bits 22:20 | `classes/clc7c0qmd.h` (`NVC7C0_QMDV03_00_RELEASE*`); rows 60, 84 |
| The dependence counter is QMD bits 157:142; the QMD table is shared by all channels | `clc7c0qmd.h`; row 84; `native_qmd.h` |
| A QMD whose shader cannot run still completes and releases | Row 61 |
| GPU VA equals CPU VA for memory registered with the GPU session | Row 30 |
| Switch 2 block-linear GOBs differ from Eden's: offset bits 5 and 6 swapped | Row 69; `eden-nx2/src/video_core/host_shaders/astc_decoder.comp` (line 1410) |
| Root tables: banks 24–31, 256 bytes each, loaded with `SET_ROOT_TABLE_SELECTOR` (0x141) and `LOAD_ROOT_TABLE` (0x142) | `classes/clc797.h`; `eden-pr4588/src/video_core/native_root_table.h`. Row 65 reads bank 24 differently: see "What to verify" |

## Threads and data flow

```text
 vCPU threads ──(gsv #11: submit, present)──▶ submission queue (MPSC, per channel)
                                                   │
                                                   ▼
                                      ┌──────── front thread ────────┐
   guest memory ◀──── reads (GP words, │ channels · host methods     │
   (copies only)       semaphores,     │ acquires · split methods     │
                       MME DMA)        │ register files · macros+memo │
                                       └──────────────┬───────────────┘
                                                      │ ring (SPSC, bounded)
                                                      ▼
                                      ┌──────── back thread ─────────┐
   guest memory ◀──── reads & writes   │ engine mirrors · dirty state │──▶ shader pipeline (#13)
                       (vertex, index, │ draws · QMD · copies · 2D    │
                       textures, QMDs, │ texture/buffer caches        │──▶ render backend (#14)
                       releases)       │ release & fence actions      │◀── completions
                                       └──────────────────────────────┘
```

| Thread | Owns | Never touches |
|---|---|---|
| vCPU threads (through gsv) | Nothing in the GPU; they call `Gpu::submit` and `Gpu::present`, which never block | Engine state, caches |
| Front | Per-channel submission queues, pusher state (current method, count, mode, position), host-method state, the per-channel register files, macro code and data RAM, the memo, acquire registrations | Caches, the render backend |
| Back | Engine mirrors and dirty flags, the QMD table, the texture and buffer caches, the open render recording, pending fence actions | Register files, macro state |
| Render submit (#14) | Host queues, swapchain, completion of host submissions | Engine state |

Every piece of state has exactly one owning thread. The only shared structures are the ring,
the submission queues, the acquire waiter table (a small mutex-guarded list) and the write-track
dirty log (see "GPU memory access").

## The front thread

### Channels and submissions

A channel is created by gsv (#11) for a GPU session and is identified by a `ChannelId`. Each
channel has its own GP entry queue, pusher state, subchannel bindings, register files and macro
data RAM. `Gpu::submit(channel, entries)` copies the 8-byte GP entries (gsv has already copied
them out of the guest's request) into the channel's queue and wakes the front. It never blocks a
vCPU thread.

**Command words are read when they are executed, never at submission.** NVN2 writes a command
buffer after submitting the GP entry that points at it, and gates it with an acquire (row 50), so
reading at submission time would run stale commands. Words are read through `GpuMemory` (below)
in chunks, as copies.

The front loop:

```text
loop
  take new submissions into their channels' queues
  pick the next runnable channel (round robin)
  run it until: its queue is empty, it parks on an acquire or a catch-up,
                or it has used its quantum of words (fairness)
  if no channel is runnable: publish the ring, then park until a submission arrives,
                the back reports a release, or the acquire poll timer fires
```

The quantum is 4,096 words (**guess**; tuned by measurement, it only affects fairness, never
correctness, since channels order themselves through semaphores).

### The method stream

The front decodes each GP entry and each method header with pure functions (`gpu::fifo`) and
keeps the pusher state per channel: method, subchannel, remaining count, mode (incrementing,
non-incrementing, increment-once) and the position in the current entry. A data word is
dispatched by method number: below 0x40 to the host methods, otherwise to the engine bound on
the subchannel.

- **Immediate** (opcode 4) carries 13 bits of data in the header itself.
- **Subdevice-mask operations** (opcodes 0 and 2) are single-GPU no-ops and are skipped.
- **End of segment** (opcode 7) ends the current entry (**guess** from the opcode's name in
  `clc76f.h`; Eden ignores it; confirm with a trace that contains one).
- **Opcode 6** is reserved: the word is a malformed-stream error (below).

### Host methods

The host methods are the `C76F` set (facts table). Bind object records the class on the
subchannel and tells the back (a `Bind` ring entry); class 0 unbinds. Subchannel 2 starts bound
to inline-to-memory, as Eden-NX2 found NVN2 expects (row 49). Releases, reductions, wait for
idle, memory operations and set reference become ring entries for the back. Acquires are handled
on the front.

### Semaphore acquires

An acquire (`SEMAPHORED` with an acquire operation, or `SEM_EXECUTE` with an acquire operation)
reads the 32-bit word at the semaphore address and compares it with the payload:

| Operation | Met when |
|---|---|
| `SEMAPHORED` acquire (1) | `value == payload` |
| `SEMAPHORED` acquire-greater-or-equal (4) | `value >= payload` |
| `SEMAPHORED` acquire-and (8) | `value & payload != 0` |
| `SEM_EXECUTE` acquire (0) | `value == payload` |
| `SEM_EXECUTE` strict-greater-or-equal (2) | `value >= payload` |
| `SEM_EXECUTE` circular-greater-or-equal (3) | `(value - payload) as i32 >= 0` |
| `SEM_EXECUTE` and (4) / nor (5) | `value & payload != 0` / `== 0` |

The conditions are those of `eden-nx2/src/video_core/engines/puller.cpp` (`PendingAcquire::Matches`,
`ProcessSemaphoreExecute`), with the operation values from `clc76f.h`.

When the acquire is met, the channel continues. When it is not:

1. The front **registers a waiter** (channel, address, operation, payload) in the shared waiter
   table, *then* reads the word again. Registering first means a release the back performs in
   between is either seen by the second read or latches the waiter (row 70).
2. If still unmet, the channel **parks at the acquire's word**. Its pusher state already points
   past the acquire; the rest of the entry is read again on resume, because the CPU may have
   written it since (row 50).
3. A parked channel is rechecked when: the back reports it performed a release (it reports
   every release, and says whether it latched a waiter); a new submission arrives on any
   channel; or the acquire poll fires, for semaphores the CPU releases (decision 3). A latched
   waiter resumes at once, even if the word has moved on since.

The front **never waits for the back to drain** to check an acquire. Eden-NX2's front drains the
back on every acquire, about 2,400 times a second (row 142). Here a release still in the ring
simply has not happened yet: the channel parks, and the back's report wakes it. The cost is a
little latency on that channel only.

Eden-NX2 never suspends on a `SEM_EXECUTE` acquire; it checks once and continues
(`ProcessSemaphoreExecute`). This design suspends on both kinds, because an acquire that does not
wait is not an acquire. Whether NVN2 uses `SEM_EXECUTE` acquires at all is to be traced
("What to verify").

A release whose address lies in the syncpoint shim range that gsv reports (row 46) is passed to
gsv as a syncpoint operation instead of a memory write (**guess** about how increments reach
syncpoints on this GPU; confirm with a trace of NVN2's fence path).

### Split methods

Method state belongs to the channel, not to the GP entry: when an entry ends with words still
owed to a method, the next entry of the same channel continues them, whatever its address or
level (row 73). A submission that ends with a method unfinished keeps it pending for the next
submission of that channel, and this is logged once per method (Eden-NX2 drops parked methods at
the end of a submission; with continuous state there is nothing to drop). A macro call's
parameters may therefore arrive across entries; the call runs only when its last word arrives.

### Engine methods: the register file

The front keeps, per channel and per class in use, a register file: a flat `[u32; N]` indexed
by method, with N the class's method space below its macro range (0xE00 for 3D). It is the only
authoritative copy (decision 1). For each incoming engine write the front:

1. Applies the shadow RAM control (`SET_MME_SHADOW_RAM_CONTROL`, method 0x49) for 3D: in track
   modes the value is also stored in the shadow file; in replay mode the value written is the
   shadow file's (behaviour from `maxwell_3d.cpp`, `ProcessShadowRam`, row 144).
2. Applies the register's own side effects that change other registers. These are computed here
   and sent as ordinary writes, so the back never changes a register on its own: the constant
   buffer upload offset that advances with each `LOAD_CONSTANT_BUFFER` word, the macro bind
   pointers (`LOAD_MME_START_ADDRESS_RAM_POINTER` and `_RAM`), the macro upload pointer, and the
   root table offset. The list is per class and is a test fixture: Eden-NX2's verifier found
   three such cases it had missed (row 144).
3. **Drops the write** if the method is a state method and the value equals the register's.
   Every other write updates the file and becomes a ring entry. A method is a state method only
   if the class table says so; anything unknown is treated as a trigger and always sent, so a
   wrong table costs speed, not correctness (decision 12).

Runs of data words to the same trigger method (constant buffer loads, inline-to-memory data,
inline indices, inline QMD data) go into one ring entry with their words, not one entry per word.

### Macros

Macro code and start addresses are uploaded with the `LOAD_MME_*` methods (0x45–0x48). The
front holds the code and the 3D and compute engines' start tables; data RAM is per channel (row
84) and shared by that channel's 3D and compute classes. Eden-NX2 keeps one code store for all
of a process's channels (row 53); this design does the same, and records uploads per channel so
the choice can be checked (decision 11).

A call runs the existing interpreter (`crates/gpu/src/mme/interp.rs`) with the front as its
`MacroHost`:

- `load` reads the call's parameters, then any words a DMA read appended;
- `state` reads the front's register file (after shadow RAM rules), which is exact because the
  front owns it;
- `method` feeds the write back into the same pipeline as a stream write (shadow RAM, side
  effects, same-value drop), in order;
- `barrier` performs pending MME DMA reads.

**A run's writes are buffered and committed only when it finishes.** That gives three things:
the memo can store them; a run that fails (`MacroError`) commits nothing and is reported once per
macro with its parameters; and a run that needs a catch-up can be retried. The step limit stays
`StepLimit::DEFAULT`.

**MME DMA reads** read guest memory the GPU may have written (query counters, row 58; indirect
arguments, row 120). When a run reaches a barrier with a DMA read pending and the back has
unperformed writes in the ring, the run is abandoned with nothing committed, the channel parks
at the macro call ("catch-up"), and the back reports when everything sent before it is performed.
The call is then run again. If the back has nothing pending, the read happens at once.

**The memo** (port of `eden-nx2/src/video_core/macro_memo.h`) sits in front of the interpreter.
A call qualifies when it read only its parameters and data RAM words: no register reads, no DMA
reads. Its key is the macro start and the parameters; its guard is the data RAM words it read;
its record holds its data RAM writes and its method writes. A hit replays the record. The code
store's generation clears the memo when code is uploaded. Eden-NX2 replays 97% of calls this way
(row 117). The interpreter is the memo's oracle in tests.

### Malformed streams

Guest command buffers are untrusted. Nothing here panics; every error is an `Err` with a
variant, counted and logged once per kind and channel:

- a GP entry or command word at an unmapped address, or an entry longer than the class allows:
  the channel stops with a channel error, as hardware raises one, and gsv is told (it can report
  it to the guest if the protocol has a way, or the run reports it);
- an unknown host method, a method on an unbound subchannel (row 49), opcode 6: the word is
  dropped and logged;
- a macro error: the call commits nothing and is logged.

Under the headless contract's strict policy, any of these ends the run with exit code 4
(`observability.md`). Decision 13 covers the default.

## The ring

### Entries

The ring is a fixed-size buffer of 64-bit words with typed entries encoded in them (the encoding
is internal; a text form exists for tests and captures). Sketch, names provisional:

```rust
enum RingEntry<'a> {
    Bind { channel: ChannelId, subchannel: u8, class: ClassId },
    Write { channel: ChannelId, class: ClassId, method: u16, value: u32 },
    Run { channel: ChannelId, class: ClassId, method: u16, words: &'a [u32] },
    Release(Release),                 // semaphore release or reduction, from any engine
    Syncpoint { id: SyncpointId },    // increment, from a submission's fence or the shim
    WaitForIdle { channel: ChannelId },
    Invalidate { channel: ChannelId, what: Invalidation },
    Marker { token: MarkerToken },    // the back reports when it has performed everything before it
    Present(PresentRequest),
    CaptureControl(CaptureCommand),
}

struct Release { address: u64, payload: u64, size: ReleaseSize, op: ReleaseOp, timestamp: bool }
```

There are no closures in the ring (Eden-NX2's ring carries function pointers,
`eden-nx2/src/video_core/gpu_split.h`): plain data can be captured, printed and replayed.

### Back-pressure

The ring holds 1 Mi words (8 MiB, Eden-NX2's size; **guess** that it is enough, to be measured
as peak occupancy). The front publishes in batches (every 256 words, or when it parks) and waits
for space when the ring is full; that waits only the front, never a vCPU. Submission queues are
unbounded lists of GP entries, but the guest bounds them itself: it paces on present (row 55)
and on its own semaphores. A queue longer than 65,536 entries (**guess**) is reported in the
flight recorder; nothing is dropped.

### Ordering guarantees

1. Within a channel, the back applies entries in that channel's method order.
2. Across channels, the back applies entries in the order the front emitted them. The guest can
   only rely on cross-channel order through semaphores, and a parked channel emits nothing.
3. A release or syncpoint increment is performed after every entry before it in the ring has
   been recorded, and after the host GPU has completed the recorded work before it (decision 2).
4. Releases and increments are performed in ring order. Completion processing is first in,
   first out, even when host submissions complete out of order.
5. A `Marker` is reported after everything before it is performed, including fenced releases.
6. A `Present` is ordered after everything emitted before it.

## The back thread: engine state

### Mirrors and dirty flags

The back keeps, per channel and class, a mirror of the register file, written only by ring
`Write` and `Run` entries, and a dirty bit set over **state groups**. A generated table maps each
method to its groups (viewports, scissors, blend per target, colour target j, depth target,
vertex stream j, vertex attributes, index buffer, constant buffer bindings per stage, texture
header and sampler pools, program addresses per stage, rasterizer state, depth and stencil
state, and so on). This is Eden's dirty-flag model (`eden/src/video_core/dirty_flags.h`), with
groups named after Mesa's methods. A write marks its groups; deriving a draw clears them.

Typed accessors read fields out of the mirror (`state.viewport(i).scale_x()`); they are
generated from the same tables (decision 6) and do bounds and enum checks, returning `Err` for
values the class header does not define.

### 3D: from state to a draw

A draw method (`DRAW_VERTEX_ARRAY_BEGIN_END_A/B`, `DRAW_INDEX_BUFFER_BEGIN_END_A/B`,
`DRAW_INLINE_INDEX`, the inline index forms, and the begin/end pairs) triggers derivation:

1. For each dirty group, re-derive its part of the draw description and keep it. Clean groups
   reuse the previous part, so a draw that changed one constant buffer binding rebuilds only that.
2. Resolve programs: for each enabled stage, the program address (`SET_PROGRAM_REGION` plus
   `SET_PIPELINE_PROGRAM_ADDRESS(j)`) is looked up in the shader cache; a miss asks the shader
   pipeline to translate it (see #13 below). The translated shader says which constant
   buffers, textures, samplers, images and storage buffers it uses, and where bindless handles
   come from.
3. Resolve resources: constant buffers from `BIND_GROUP_CONSTANT_BUFFER(j)` (and root tables for
   banks 24–31), textures and samplers from the header and sampler pools by handle, vertex and
   index buffers, render targets. Each becomes a host resource through the buffer and texture
   caches.
4. Emit a `Draw` command (see "What the back hands the render backend").

A draw whose shader was rejected is skipped and counted, once per program in the log (negative
cache, rows 90, 112). Conditional rendering (`SET_RENDER_ENABLE_*`) is evaluated here from the
query value in guest memory, after a catch-up when that value is still pending.

### Compute: QMD 3.0

NVN2 dispatches compute through QMDs (row 53). The front sends the inline QMD words and the PCAS
methods as ordinary writes; the back:

1. On the last `LOAD_INLINE_QMD_DATA` word, writes the 256-byte inline QMD to guest memory at
   `SET_INLINE_QMD_ADDRESS` and, if `SET_QMD_VIRTUALIZATION_CONTROL` (method 0xA7) has
   `SEND_PCAS_ENABLE` (bit 24) set, schedules it (`clc7c0.h`; Eden-NX2's `kepler_compute.cpp`
   also schedules when bit 12 of method 0xCC is clear, a condition no header explains, kept as a
   listed divergence).
2. Keeps **one QMD table for all channels** (row 84): per QMD address, whether it is loaded,
   scheduled, completed, and its dependence counter plus deferred adjustments. `SEND_PCAS_A` and
   `SEND_SIGNALING_PCAS2_B` actions schedule (2, 3), decrement (7) or increment (12) a QMD; a QMD
   runs when it is loaded, scheduled, not completed and its counter is zero. The logic and its
   limits (4,096 QMDs, chains of 256) are those of `eden-nx2/src/video_core/native_qmd.h`, each
   an `Err` here, not a silent stop.
3. A QMD that runs is read again from guest memory first, because NVN2 patches its release
   payloads while it waits (comment in `native_qmd.h`). It becomes a `Dispatch` command: program
   address (words 48–49), grid (words 12–14), block (words 18–19), shared memory, local memory,
   and the eight constant buffers (words 32–47), as `DispatchNativeCompute` reads them.
4. After the dispatch is recorded, each enabled release slot (24, 28, 52) becomes a fenced
   release with its reduction, and a dependent QMD action is applied. A QMD whose program was
   rejected still completes and releases (row 61).

Row 62 says a release whose flags have bit 19 set must write `max(old, payload)`. In Mesa's QMD
3.0 header that bit is `RELEASE0_MEMBAR_TYPE`, and a max reduction is reduction enable (bit 26)
with op 2. This design implements the header's reduction fields, plus Eden-NX2's bit-19 rule as
a listed divergence pinned by a test, until a trace settles it ("What to verify", decision 10).

### DMA copy, inline-to-memory and 2D

- **DMA copy (`C7B5`).** `LAUNCH_DMA` (byte 0x300) with the source and destination layouts
  (pitch or block-linear, bits 7 and 8), multi-line, remap and semaphore fields from
  `classes/clc7b5.h`. When either side overlaps an image the texture cache holds, the copy is a
  host copy command; otherwise the back copies through guest memory, using the layout crate for
  block-linear sides. The destination range is invalidated in the caches. The semaphore is a
  fenced release.
- **Inline-to-memory (`A140`, and `LAUNCH_DMA`/`LOAD_INLINE_DATA` at bytes 0x1B0/0x1B4 in the 3D
  and compute classes).** The back writes the data to guest memory (pitch or block-linear) in
  order. Cache invalidations from many small writes are merged until the next draw, dispatch or
  copy (row 134).
- **2D (`902D`).** Surface-to-surface blits with scaling and filtering become host blit commands
  when the caches hold either surface, and CPU copies through the layout crate otherwise. Eden's
  software blitter cost 3.6% of the GPU thread on small mip copies (row 137), so the host path is
  the default. Which 2D class NVN2 binds is not in the notes: `902D` is a **guess** (no Ampere 2D
  class exists in Mesa's headers); a bind trace confirms it.

### Releases and completion

The back records commands into an open host submission. A fenced action (release, syncpoint
increment, marker) is attached to the open submission if it has recorded work since the last
submission was closed; otherwise to the newest submission still in flight; if nothing is in
flight, it is performed at once (Eden-NX2's empty-fence skip, row 123).

The open submission is closed and handed to the render backend when: the ring is empty, a
present arrives, a wait for idle arrives, a fenced action's address has a registered waiter (a
channel is parked on it), or 1 ms of recording has passed (**guess**, so that a CPU spinning on a
semaphore is not starved; measure).

When the render backend reports a submission complete, the back performs its actions in order:
writes the release (4 or 16 bytes, with the timestamp for 16-byte releases, row 52), or applies
the reduction; latches any registered waiter the new value satisfies; tells gsv about syncpoint
increments; reports markers. Then it wakes the front if anything was latched or reported.
Completions are batched: one wake for all actions of all submissions completed at that point
(row 124).

Release writes happen on the back thread only, so their order is the ring's order.

## GPU memory access and cache coherence

### Reading and writing guest memory

The GPU reads and writes through `GpuMemory`, a trait this crate defines and gsv implements for
a GPU address space (#11). GPU addresses equal CPU addresses for memory registered with the
session (row 30); GPU-only regions (sparse regions the graphics server backs, rows 57, 80, 85)
are resolved by the same implementation. Following the execution model, access is by copy or
through the memory crate's raw view type, never a Rust slice into guest memory.

```rust
pub trait GpuMemory: Send + Sync {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault>;
    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault>;
    /// Raw view for hot paths (command words, large uploads), as in the execution model.
    fn view(&self, va: u64, len: u64) -> Result<RawView<'_>, GpuFault>;
    /// Bumped by every map, unmap or kind change, for per-thread translation caches.
    fn generation(&self) -> u64;
}
```

Each GPU thread keeps its own translation cache, invalidated by the generation counter. A cache
shared between threads crashed in Eden-NX2 when a service thread remapped memory (row 125).

### Learning of CPU writes

The texture and buffer caches use the execution model's write tracking: pages they upload from
are protected read-only in stage 1; the first CPU write faults, the kernel restores write access
and records the page as dirty. This crate cannot depend on the kernel (it sits above `gsv` in the
dependency order), so the seam is a type in `starquake-memory`, which both depend on:

```rust
pub struct WriteWatch { .. }    // shared: Arc<WriteWatch>
impl WriteWatch {
    pub fn request(&self, range: PageRange) -> WatchTicket;   // back thread: please protect
    pub fn armed(&self, ticket: WatchTicket) -> bool;         // has protection taken effect?
    pub fn take_dirty(&self, out: &mut Vec<PageRange>);       // back thread: drain
}
```

The kernel side (applying protections, which needs a TLB flush on some core, and recording
faults) is the kernel's; what this design assumes of it is listed under "Interfaces". Until a
watch is armed, the cache treats the range as untracked and reads it again at each use. The back
drains the dirty log before each draw, dispatch and copy; with nothing dirty, that is one atomic
load.

A range the guest is known to rewrite constantly (streamed uniform data, row 141) costs one
fault per page per tracking period. The caches may mark a range "streamed" after repeated faults
and stop tracking it, re-reading it each use instead.

### GPU writes and the CPU

Where the render backend can alias guest memory (unified memory, `MTLStorageModeShared`), GPU
writes to buffers land in guest memory directly. Images are stored in host layouts and are not
written back by default (decision 4). They are written back, through the layout crate, when a
DMA copy or inline write reads them as plain memory, when an MME DMA read touches them, and on
request in tests. Eden-NX2 measured that guest CPUs read GPU-written buffer memory at most 64
times in 200 seconds, while downloading at every fence cost 250–330 MB/s (row 131).

### Caches

The buffer cache and the texture cache live in this crate, on the back thread. They map guest
ranges to host resources created through the render backend, and own invalidation: CPU writes
(dirty log), GPU writes (copies, inline writes, render targets) and unmaps (generation). Texture
headers (`classes/clc797tex.h`) are decoded here into an image description handed to the layout
crate and the render backend. Unmapped GPU addresses get a stand-in identity fixed by the GPU
address, so an image is found again (row 103). Eviction thresholds come from the render backend's
memory budget (rows 119, 139).

## Block-linear layout: `starquake-layout`

Image layout is pure arithmetic over a description: format, extent, levels, layers, sample
layout, GOB kind, block height and depth. It decides level offsets, mip tails, layer strides and
total size, and converts between linear and tiled bytes for a region. It is ported from Mesa's
`nil` crate (`mesa-nouveau/src/nouveau/nil/{tiling,image,extent,format}.rs`, MIT) into a new leaf
crate with no workspace dependencies, in safe Rust (nil's copy routines use raw pointers;
`copy.rs`), operating on byte slices the caller has already copied out of guest memory.

Three users need it: the back (CPU copies, 2D fallback, write-back), the render backend (a GPU
compute deswizzle must agree with it), and gsv (sizes for its native image and window requests,
rows 33, 47, 96). A leaf crate is the only place all three can reach (decision 5).

**The Switch 2 GOB is Mesa's Turing colour GOB.** Eden's GOB swizzle (`SwizzleOffset` in
`astc_decoder.comp`) puts byte offset bits 0–3 = x bits 0–3, bit 4 = y bit 0, bit 5 = x bit 4,
bits 6–7 = y bits 1–2, bit 8 = x bit 5. Within a 512-byte GOB, 32-byte sector number
`x4 + 2·y1 + 4·y2 + 8·x5` gives the layout Mesa calls `TegraColor`. Eden-NX2 swaps offset bits 5
and 6 for Switch 2 (row 69), giving sector `y1 + 2·x4 + 4·y2 + 8·x5`, which is exactly Mesa's
`TuringColor2D` table (`tiling.rs`), the GOB Mesa chooses for colour on Turing and later. Depth
and stencil use `FermiZS` on Turing in Mesa; whether NVN2 does is not in the notes (**guess**,
confirm with a depth target dump). The derivation is checked offline by the layout crate's first
test, which compares both swizzles bit for bit.

Offline tests: golden offsets hand-computed from the sector tables above; tile-then-untile round
trips over generated extents and formats; Eden's `SwizzleOffset` as a second oracle for the
Tegra GOB; property tests that no description panics or overflows (huge images, row 87).

## What the back hands the render backend

The back produces a stream of plain-data commands and hands it over in submissions. Sketch,
names provisional; the render backend spec (#14) owns the final types:

```rust
enum RenderCommand {
    BeginPass(PassDesc),          // colour targets (image views), depth target, render area, load ops
    EndPass,
    Draw(DrawDesc),
    Dispatch(DispatchDesc),
    Copy(CopyDesc),               // buffer/image, either direction, with regions
    Blit(BlitDesc),               // 2D engine: scaled, filtered
    Clear(ClearDesc),
    Upload(UploadDesc),           // bytes the back read from guest memory, already linear
    Barrier(BarrierDesc),
}

struct DrawDesc {
    pipeline: PipelineKey,        // shader ids per stage + fixed-function state, hashable
    bindings: StageBindings,      // per stage: uniform buffers, textures+samplers, images, storage
    vertex: VertexInput,          // streams (buffer slice, stride, divisor) and attributes
    index: Option<IndexInput>,
    params: DrawParams,           // direct (first, count, base vertex, instances) or indirect
    dynamic: DynamicState,        // viewports, scissors, blend constants, stencil refs, depth bias
}

struct Submission { commands: Vec<RenderCommand>, id: SubmissionId }
```

Every value in a command is already decoded from guest registers and validated. A host resource
appears only as a typed id from the render backend. A recording implementation of the backend
turns a submission into text, which is what the offline tests snapshot.

## Capture and replay

The observability spec plans GPU capture (`observability.md`, "Later"): register traffic goes
to the GPU's own capture format, off by default. This is that format.

### What a capture holds

A capture starts at a present boundary and covers one or more frames:

1. **A header**: magic `SQGC`, a format version, the build id, and the channel list.
2. **The starting state**: every channel's register files and shadow file, pusher state,
   bindings, macro code and data RAM; the back's QMD table. Caches are emptied at capture start,
   so every resource is read during the capture.
3. **One input log per GPU thread**, each in the order that thread consumed things:
   - front: submissions taken (channel, GP entries), memory reads (address, length, content
     hash), acquire checks (value seen), back reports received, poll ticks;
   - back: memory reads, completions received, write-tracking drains.
4. **A blob table** of read contents, deduplicated by hash, so a command buffer read twice is
   stored once.
5. **The outputs**, for comparison: a hash per ring entry, the render command stream, and the
   writes the back made to guest memory.

Recording reads in the order each thread made them, rather than a memory snapshot at frame
start, is what makes gated command buffers replay correctly: the CPU writes them after
submission, between the GPU's reads (decision 8).

### Replay

`starquake gpu-replay <capture> --until present=N --dump ring|render|writes` builds a `Gpu` with
a replay `GpuMemory` that answers reads from the log, runs the real front on its log, then the
real back on the resulting ring and its log, with the recording render backend and a shader
pipeline in recording mode (#13: translate, validate, no host compile). A read that is not the
next one in the log, or a ring entry whose hash differs, stops the replay with the position: a
divergence is a finding, as in the kernel replay. Single-threaded replay is deterministic on any
host.

### Fixtures and CI

Captures from titles never leave the machine (`observability.md`). Committed fixtures are built
by a capture builder in the GPU crate's `test-support` feature from hand-written method streams
(a small push-buffer encoder following Mesa's `nv_push.h`), and later from homebrew programs.
CI replays them and snapshots the ring text and the render command text with `insta`.

The capture format gets a golden-bytes test covering every record kind with distinct non-zero
values, and a header check before the body, as the test strategy requires.

## Observability

- The GPU keeps its own flight recorder: the last few thousand submissions taken, acquires
  parked and resumed (with the value seen), releases performed, macro errors, channel errors,
  presents. It is written as `gpu.txt` into the artifacts folder with the others (an addition to
  the observability spec's table, for the lead to reconcile).
- Counters, cheap enough to keep on: words parsed, writes dropped as same-value, macro calls,
  memo hits, acquires parked, catch-ups, ring peak occupancy, fenced actions and how many were
  skipped as empty. `summary.json` gets them per run.
- GPU register writes are not events in the kernel's stream (`observability.md`); acquires and
  releases are not either. The GPU's link to the event stream is through gsv: submissions and
  presents are IPC requests, and the order in which presents complete deferred replies is
  recorded there.

## Interfaces to neighbouring designs

### Graphics server sessions (#11)

What this spec assumes of gsv:

- gsv creates and destroys channels, one per GPU channel the guest opens, each on a GPU address
  space, and implements `GpuMemory` for that space (registered memory at GPU VA == CPU VA, sparse
  and GPU-only regions, kind changes, row 149). It bumps the generation on every change.
- gsv copies GP entries out of the guest's submit request and calls `Gpu::submit` from the vCPU
  thread handling the request. If a submit asks for a fence (an increment of the channel's
  syncpoint, row 32), gsv passes it and gets the value it will reach.
- gsv owns syncpoints: it implements `SyncpointSink`, which the back calls (from the back
  thread) when an increment is performed, and it tells the GPU the shim page range (row 46).
- gsv answers vi2 present (229) and paces the guest on it (rows 55, 59). It calls
  `Gpu::present` after the frame's fence is reached, or with the fence for the GPU to wait on.

What this spec offers gsv (sketch):

```rust
impl Gpu {
    pub fn new(deps: GpuDeps, config: GpuConfig) -> Result<Gpu, GpuError>;   // starts both threads
    pub fn create_channel(&self, memory: Arc<dyn GpuMemory>) -> Result<ChannelId, GpuError>;
    pub fn destroy_channel(&self, channel: ChannelId);
    pub fn submit(&self, channel: ChannelId, entries: &[GpEntry], fence: Option<FenceRequest>)
        -> Result<Option<SyncpointValue>, SubmitError>;                          // never blocks
    pub fn present(&self, request: PresentRequest);   // image address + layout + crop + fence
    pub fn channel_error(&self, channel: ChannelId) -> Option<ChannelError>;
}
pub trait SyncpointSink: Send + Sync {
    fn increment(&self, id: SyncpointId);
}
```

Thread rules: `submit` and `present` are called on vCPU threads and only enqueue. `SyncpointSink`
is called on the back thread and must not block (gsv wakes its waiters through kernel handles,
as the execution model's active services do). Present waiting with a bounded timeout is gsv's.

### Shader pipeline (#13)

What the back asks of the shader pipeline:

- `translate(request, env) -> Result<Arc<TranslatedShader>, Rejected>`, called synchronously on
  the back thread, in linear time (row 121). The request names the stage and the program's
  guest address. The environment is implemented by the back and valid only during the call: it
  reads program words (copies), constant buffer words (for bindless handles and jump tables),
  and texture types by handle. The translator returns what it read from the environment, and
  those reads become part of the cache key, checked again on lookup (Eden's
  `shader_environment.cpp` model).
- A `TranslatedShader` carries a `ShaderId` and its resource usage: constant buffers by bank and
  range, textures, samplers, images and storage buffers by binding or bindless source, inputs
  and outputs, local and shared memory sizes. The back needs this to bind resources.
- `Rejected` carries a reason and is cached negatively until the program's memory is written
  (rows 90, 112).
- Whether host pipeline compilation is asynchronous, and what a draw does when its pipeline is
  not ready, is the pipeline's and the backend's (#13, #14); decision 9 records what this design
  recommends.

What the back offers: the environment, program invalidation when the code's memory is written
(dirty log), and the console-provided root table contents for banks 24–31.

### Render backend (#14)

What the back assumes of the render backend, used only from the back thread:

- resource creation and destruction by description (buffers, possibly aliasing a guest range on
  unified memory; images with format, extent, levels, layers; samplers), returning typed ids;
- `submit(Submission)` that hands a recorded command list to the render submit thread, and a
  completion signal per submission that the back can wait on together with the ring;
- a memory budget the caches read for eviction;
- `present(PresentFrame)` with a host image id, crop and transform;
- a recording implementation behind its `test-support` feature, which prints submissions as text.

What the back offers: the command stream above, fully decoded; host layouts produced by the
layout crate; and the rule that every resource referenced by a submission stays alive until that
submission completes (the back defers destruction until then).

Ownership: the back owns which guest ranges map to which host resources; the render backend owns
the host objects and host memory.

### Approved specs this design leans on

These need nothing new from the approved specs except the write-track seam, which the lead
should reconcile with the kernel's port plan:

- **Execution model:** guest memory by copy or raw view (followed); stage-1 write tracking
  (followed). New: a `WriteWatch` type in `starquake-memory`; the kernel applies protection
  requests from the GPU's back thread at the next kernel entry on any core (it cannot flush the
  TLB itself), records the faulting page as dirty, and restores access.
- **Observability:** GPU capture in its own format (this spec); a `gpu.txt` artifact and GPU
  counters in `summary.json` (additions).
- **Test strategy:** "recorded method streams replayed into the front end, compared against
  expected engine state" is the capture replay above; Mesa's simulator tests stay the macro
  oracle.
- **Architecture:** adds the leaf crate `starquake-layout` (decision 5).

## Port slices

Each slice is one pull request with its own tests. "Offline" means no game, no GPU and no
hypervisor. The wave letter says what can run in parallel: every slice in a wave depends only on
slices of earlier waves (and on sibling-spec types where noted), so slices in one wave run at the
same time.

| # | Slice | Depends on | Ported from | Offline test | Wave |
|---|---|---|---|---|---|
| G1 | `gpu::fifo`: GP entry and method header decoding, the push-buffer encoder for tests | none | `classes/clc76f.h`; `eden-nx2/src/video_core/dma_pusher.h` | Hand-laid words per field from the header's bit ranges; property test: any word decodes or errors, never panics | A |
| G2 | `gpu::sema`: acquire predicates, release sizes, reductions (both host forms, MME and QMD reductions) | none | `clc76f.h`, `clc7c0qmd.h`, `clc797.h` (`MME_DMA_REDUCTION`); `puller.cpp` | Table per operation, boundary values on both sides (wrap for circular GEQ) | A |
| G3 | Class tables: a generator (`cargo xtask gen-classes`) reading Mesa's class headers into Rust tables (methods, fields, enums, array strides), committed output for `C76F`, `C797`, `C7C0`, `C7B5`, `A140`, `902D`, QMD 3.0, texture header | none | `classes/*.h`; Mesa's `class_parser.py` as the model | Golden table entries checked by hand against the header text for a sample of methods; regeneration is byte-identical | A |
| G4 | `gpu::ring`: the SPSC ring, entry encoding, text form, back-pressure, publish and park | none | `eden-nx2/src/video_core/gpu_split.{h,cpp}` (structure only) | Encode/decode round trip for every entry kind; ordering and wrap tests with two threads and a fixed schedule; a full-ring test | A |
| G5 | `starquake-layout`: GOB kinds, block-linear offsets, image layout, mip tails, tile/untile of regions | none | `mesa-nouveau/src/nouveau/nil/{tiling,image,extent,format,copy}.rs` (MIT) | The GOB derivation test above; hand-computed offsets; round trips; overflow property tests | A |
| G6 | MME memo | existing `gpu::mme` | `eden-nx2/src/video_core/macro_memo.h` | The interpreter as oracle: generated programs and calls, memo replay equals a fresh run; guard failure when data RAM changes | A |
| G7 | `GpuMemory` trait, test implementation, per-thread translation cache with generation; `WriteWatch` in `starquake-memory` (the type and its fake; the kernel side is a kernel slice) | none | Row 125; execution model | Generation invalidation; fake watch arming and dirty drain | A |
| G8 | `gpu::channel`: per-channel pusher state machine with host methods, acquire parking and resume at the exact word, split methods across entries and submissions, malformed-stream errors | G1, G2, G7 | `dma_pusher.cpp`, `puller.cpp` | Streams built with G1's encoder: suspend mid-entry and resume re-reading changed words; split method across entries; every error variant asserted | B |
| G9 | `gpu::regs`: front register files, shadow RAM control, side-effect registers, same-value dropping, run batching | G3 | `maxwell_3d.cpp` (`ProcessShadowRam`, `CallMethod`), `kepler_compute.cpp` | Shadow RAM modes; each side-effect register; a same-value write dropped, a trigger never dropped; unknown methods pass | B |
| G10 | `gpu::back::state`: mirrors, dirty groups, typed accessors for 3D and compute | G3 | `eden/src/video_core/dirty_flags.{h,cpp}` (model) | Each method marks its groups; accessor bounds and enum errors | B |
| G11 | `gpu::back::copy`: DMA copy, inline-to-memory, 2D (CPU paths) | G3, G5, G7 | `eden-nx2/src/video_core/engines/{maxwell_dma,engine_upload,kepler_memory,fermi_2d}.cpp` | Copies between pitch and block-linear in a fake memory against G5; inline writes; semaphore fields | B |
| G12 | `gpu::back::qmd`: QMD 3.0 decode, PCAS scheduler, release slots, dependent QMDs | G2, G3 | `native_qmd.h`, `kepler_compute.cpp` (`ProcessNativeMethod`, `DispatchNativeCompute`) | Golden QMD words from header bit positions; the cross-channel decrement-before-schedule case (row 84); rejected QMD still releases (row 61); every limit as an error | B |
| G13 | Front macro host: uploads, start tables, calls across entries, per-channel data RAM, buffered writes, MME DMA reads with catch-up, memo hook | G6, G8, G9 | `maxwell_3d.cpp` (`ProcessMacro`, `CallMacroMethod`), `kepler_compute.cpp`, row 58 | Recorded-call style tests with a fake back; a DMA read that forces a catch-up commits nothing on the first run | C |
| G14 | `gpu::front`: the front loop, channel scheduling, waiter table, parking and wake-ups, acquire poll | G4, G8, G9, G13 | `gpu_split.cpp`, `nx2.h` (`Nx2SemaphoreWaiters`) | Scripted multi-channel scenarios with a fake back and fake clock: a parked channel does not block another; latching an equal-acquire whose value moves on (row 70) | D |
| G15 | `gpu::back::fence`: open submission, fenced actions, empty-fence skip, completion batching, release writes, latching, syncpoint sink calls | G2, G4, #14's submission and completion types | Rows 123, 124; `kepler_compute.cpp` (release callbacks) | Fake render backend completing out of order; actions still FIFO; empty fences skipped | C |
| G16 | Buffer cache: guest range to host buffer, write-watch integration, streamed ranges, eviction by budget | G7, #14's resource types | `eden/src/video_core/buffer_cache` (reference only); rows 120, 131, 139 | Fake memory, fake watch, recording backend: CPU write re-uploads once; unmap drops | C |
| G17 | Texture cache: texture header decode, image identity by address and layout, uploads through G5, render targets, write-back on demand | G5, G7, G10, #14's resource types | `eden/src/video_core/texture_cache` (reference only); rows 69, 87, 103 | Recording backend: same header twice is one image; CPU write invalidates; write-back matches G5 untile | C |
| G18 | `gpu::back::draw`: derive `DrawDesc` from 3D state, shader requests, bindings, conditional render | G10, G16, G17, #13's request types | `eden-nx2/src/video_core/engines/{maxwell_3d,draw_manager}.cpp` (reference) | Register scripts to golden draw descriptions (`insta`), with a fake shader pipeline returning fixed usage | D |
| G19 | `gpu::back` executor: ring entries to engines, compute dispatch emission, present | G10, G11, G12, G15, G18 | none (new) | Ring text in, render command text out (`insta`) | E |
| G20 | Capture writer and reader, replay memory, `gpu-replay` command, capture builder in `test-support` | G14, G19 | none (new) | Golden bytes for every record kind; a built capture replays to its expected outputs; a tampered read is reported as a divergence | F |
| G21 | Wiring: `Gpu` handle, threads, gsv submission and present path, counters, flight recorder, `gpu.txt` | G14, G19, #11's session types | none (new) | A homebrew-style synthetic stream submitted through a fake gsv produces the expected render commands | F |

Notes on scheduling:

- **Wave A (G1–G7) is seven independent slices**, each in its own module or crate. They are
  well-specified ports and suit a parallel wave.
- G8 (the channel state machine), G13 (the macro host) and G14 (the front loop) carry the
  design's subtle parts (exact-word resume, catch-ups, latching). They are judgement-heavy and
  should go to one agent in sequence, as the working mode prefers for foundations.
- G16, G17 and G18 depend on the render backend's and shader pipeline's types. If those specs
  land later, the slices start against provisional types in a `pending` module and are adjusted
  when the sibling specs settle.
- The kernel half of `WriteWatch` (applying protections, recording faults) is a kernel slice,
  not listed here.

## What to verify before building

Measurements and traces that could change the design, most consequential first. Traces from
titles stay local; what they settle is written down as facts with the trace described.

1. **QMD release flag bit 19.** Row 62 treats it as a max release; Mesa's QMD 3.0 header calls
   it `RELEASE0_MEMBAR_TYPE`. Dump the release words of QMDs a title schedules: if max releases
   always carry reduction enable (bit 26) with op 2, drop the bit-19 rule. Changes G12.
2. **How often a channel parks, on what, and for how long**, split between semaphores the CPU
   releases and those the GPU releases. This decides whether the acquire poll is good enough or
   CPU writes to semaphore pages should wake the front (decision 3). Changes G14.
3. **How often MME DMA reads happen** and whether the data they read is GPU-written. Each one
   with pending back work costs a catch-up. If they are frequent, the front needs a cheaper way
   (for example, letting the back answer the read). Changes G13.
4. **Whether macro code is uploaded per channel or once per process** (row 53 shares it). Trace
   `LOAD_MME_*` per channel. Changes G13 (decision 11).
5. **Whether NVN2 uses `SEM_EXECUTE` acquires**, and whether any must not block (Eden-NX2 never
   blocks on them). Changes G8.
6. **Bank 24 and the root tables.** Row 65 synthesizes a console-provided blit block in bank 24;
   PR #4588 models banks 24–31 as root tables loaded through methods 0x141/0x142. Trace
   `LOAD_ROOT_TABLE` writes: if NVN2 loads bank 24 itself, the synthesis is not needed. Changes
   G18 and #13.
7. **The release mirror page** (row 35: releases into page P+0x1000 mirrored to page P, a guess
   in Eden-NX2). Find what really copies them (the syncpoint shim, row 46, is a candidate).
   Changes G15 and #11.
8. **Ring occupancy and front/back balance** on a real stream (peak words in the ring, time the
   front waits for space, time the back waits for work). Changes the ring size and the batching
   constants.
9. **The depth GOB kind** and the end-of-segment opcode (both marked guesses above): a depth
   target dump and a stream containing opcode 7.
10. **Host class and 2D class ids** NVN2 binds, from a bind trace.
11. **Memo hit rate with per-channel data RAM** in this interpreter, to confirm Eden-NX2's 97%
    (row 117) carries over before considering a JIT.

## Decisions for the owner

Each decision stands on its own. The recommendation comes last in each.

1. **Who keeps the GPU's register values.** The GPU's engines are configured by writing
   numbered registers; macros (small GPU programs) read them back. Options: (a) the front thread
   keeps the only copy and sends the back each change; (b) the back keeps them and macros run on
   the back, as Eden-NX2's first split did; (c) both keep copies and a checker compares them, as
   Eden-NX2's final split does (row 144). (a) is simplest and fastest but requires every rule
   that changes a register to be computed on the front; (b) is safe but leaves the front with
   little to do, losing most of the split's gain; (c) is fast but carries two copies that can
   drift. **Recommended: (a).** Eden-NX2 found the macros-on-the-front arrangement the
   fastest, and making the front the single owner removes the drift its checker had to catch.
2. **When a GPU "done" signal becomes visible to the game.** The GPU tells the game it has
   finished work by writing a value (a semaphore release). Options: (a) write it as soon as the
   back thread has *recorded* the work; (b) write it once the host GPU has actually *finished*
   it. (a) is faster but lets the game reuse memory the host GPU is still reading or read
   results not yet written; (b) is correct but adds latency and a fence per release, which
   Eden-NX2 had to batch and skip when empty to keep cheap (rows 123–124). **Recommended: (b),
   with the empty-fence skip and batching**, and (a) available as a debugging switch. Wrong
   results are much harder to diagnose than slow ones.
3. **How the GPU notices the game's CPU releasing a semaphore.** A channel waiting on a value the
   CPU will write has to notice the write. Options: (a) re-read waiting semaphores on a short
   timer while anything waits (spin briefly, then sleep in growing steps up to about 100 µs);
   (b) write-protect the semaphore's page so the CPU's write traps and wakes the GPU at once.
   (a) is simple and needs nothing from the kernel but adds up to the poll interval of latency
   and some idle CPU; (b) wakes instantly but costs a trap per write and a re-protection each
   time, and depends on the kernel's write-tracking path. **Recommended: (a) first**, measured
   (verify item 2), with (b) added only if the measurement shows the poll costs frames.
4. **Copying GPU-drawn images back into game memory.** The host GPU keeps images in its own
   layout; the game's memory holds the console's layout. Options: (a) never copy back unless
   something on our side reads that memory as plain bytes (a copy engine, an inline write, a
   macro DMA read); (b) copy back at every fence, as Eden does by default; (c) trap the game's
   CPU reads of GPU-written pages and copy back on demand. (a) is cheapest but a game that reads
   a rendered image with the CPU sees stale data; (b) cost Eden-NX2 250–330 MB/s for almost no
   benefit (row 131); (c) is exact but read traps were expensive in Eden-NX2 (row 115).
   **Recommended: (a)**, with (c) deferred as an accuracy option. Buffers on unified memory
   need no copy at all.
5. **Where texture tiling code lives** (issue #17). Options: (a) inside the GPU crate; (b) a new
   small crate with no dependencies, `starquake-layout`; (c) inside the render backend. The GPU
   back thread, the render backend's GPU deswizzle and the graphics server's size calculations
   all need it, and (a) and (c) are each unreachable from one of them under the dependency
   rules. (b) adds a crate to the architecture. **Recommended: (b).** It is pure, ported from
   MIT code, testable offline, and the only place all three users can reach.
6. **Where register definitions come from.** Options: (a) generate Rust tables from Mesa's
   NVIDIA class headers with an `xtask` command and commit the output; (b) write register
   definitions by hand as each feature needs them. (a) gives every method name and field at once
   and is checkable against the source, but adds a generator to maintain and large generated
   files; (b) is smaller but slow, error-prone and leaves gaps. **Recommended: (a).** The headers
   come from NVIDIA's own published documentation and cover the classes exactly.
7. **Which GPU classes to model.** Options: (a) only the Ampere-era classes Switch 2 games use,
   with the class table designed so older ones can be added; (b) Ampere and the Maxwell classes
   Switch 1 games use, now. (a) is less work and keeps focus; (b) would serve backward
   compatibility sooner. **Recommended: (a)**, with Maxwell classes deferred, not ruled out
   (see "Deferred").
8. **What a GPU capture records.** Options: (a) every byte each GPU thread read, in order,
   deduplicated; (b) a snapshot of all GPU-visible memory at the start of the frame plus the
   submissions. (a) replays exactly, including command buffers the game writes after submitting
   them (row 50), but costs time while capturing; (b) is cheaper to take but cannot reproduce
   those mid-frame writes, so it replays the wrong commands for the same games we most need to
   debug. **Recommended: (a).** Capture is off by default, so its cost only matters while
   debugging.
9. **What a draw does when its shader is not ready yet.** Shaders are translated and then
   compiled by the host driver, which can take milliseconds. Options: (a) wait; (b) skip the draw
   this frame (pop-in). (a) stutters on first use but always draws correctly and makes captures
   deterministic; (b) keeps frames smooth but objects flicker in. **Recommended: (a) to start**,
   with (b) as an option later. A shader that can never be translated is always skipped and
   remembered (rows 90, 112). This decision is shared with #13 and #14.
10. **When the references disagree.** Mesa's class headers (from NVIDIA's documentation) and
    Eden-NX2's observed behaviour sometimes conflict, as with QMD release bit 19 (row 62).
    Options: (a) follow the headers; (b) follow Eden-NX2; (c) follow the headers for layouts and
    field meanings, follow Eden-NX2 where it observed behaviour the headers do not explain, and
    list each such case as a divergence pinned by a test until a trace settles it.
    **Recommended: (c).** It keeps what is known separate from what was observed to work.
11. **Whether macro programs are shared by all of a game's GPU channels.** Each channel has its
    own macro scratch memory (row 84). Options for the programs themselves: (a) one store per
    process, as Eden-NX2 does (row 53); (b) one per channel, as the hardware's per-channel state
    suggests. (a) works if the game uploads the same programs everywhere and breaks only if two
    channels use different programs in the same slot; (b) breaks if the game uploads once and
    calls from several channels. **Recommended: (a)**, because it is what is known to work, with
    uploads recorded per channel to settle it (verify item 4).
12. **Dropping register writes that change nothing.** About 60% of register writes rewrite the
    value already there (row 135). Options: (a) drop them on the front; (b) pass everything. (a)
    halves the back's work, but a register wrongly classified as plain state would lose a write
    that triggers an action; (b) is always safe. **Recommended: (a)**, with any register not
    known to be plain state treated as a trigger, and a switch to turn dropping off when hunting
    a bug.
13. **What happens on a malformed command stream.** Options: (a) skip the bad word, log it and
    carry on, as Eden does; (b) stop the channel with an error, as hardware does; (c) skip
    unknown or unbound methods but stop the channel on structural errors (unreadable memory,
    impossible lengths), and stop the whole run under the strict policy. (a) keeps games going
    past our own gaps but can run garbage; (b) is faithful but turns every missing feature into a
    hang. **Recommended: (c).**

## Deferred, not ruled out

- **A macro JIT** (AArch64 first, with `dynasmrt`; x86-64 later). Eden-NX2's JIT saved about
  15% over its memo (row 140). It slots in behind the same `MacroHost` and buffered-write rules
  as the interpreter, keyed by macro start and code generation, with the interpreter as its
  oracle on recorded calls.
- **Waking on CPU semaphore writes** through write tracking (decision 3, option b): an extra
  waker for the front; nothing else changes.
- **More back threads** (for example compute on its own). Each would get its own ring; the front
  routes by class. Cross-ring order is already only through semaphores, which the fenced
  completion path handles.
- **Maxwell and older classes** for Switch 1 titles: the class table is keyed by class id, so a
  `B197` table and its engine rules are additive (decision 7).
- **Reading back GPU-written images on CPU reads** (decision 4, option c), through read traps on
  the write-tracking machinery.
- **Queries** (occlusion counts, timestamps, pipeline statistics): report semaphores
  (`SET_REPORT_SEMAPHORE_A`–`D`) first write the values the front can know (payloads, a
  timestamp from guest time); host queries plug into the fenced completion path later.
- **Mesh and task programs** (rows 89, 97): they are bound through the ordinary program slots,
  so `DrawDesc` already carries them; the shader pipeline and render backend decide how to run
  them.
- **Transform feedback, zcull, compression kinds** (row 149 only needs the kind accepted), and
  **sparse and huge virtual textures** (row 87): later texture cache and backend work.
- **GPU state in snapshots** (`observability.md`): register files, macro state, the QMD table
  and mirrors are plain data and serialisable from the start; caches are rebuilt from memory on
  resume, as the capture start already does.
- **Video decoding** (`host1x`, rows 105, 148) is gsv's and the host's; it does not go through
  this command stream.
