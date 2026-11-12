# Graphics server: sessions, syncpoints, GPU memory and display

Status: approved by the owner, 2026-11-12 (issue #11).

Every item in "Decisions for the owner" carries the owner's ruling (`docs/decision-log.md`,
2026-11-12), and the text follows them and the seam resolutions of the graphics review guide
(`docs/specs/graphics-review-guide.md`).

Builds on the execution model, the kernel spec, the service framework and the observability spec.
Three sibling specs were drafted at the same time: the GPU front/back protocol (#12), the shader
pipeline (#13) and the render backend (#14). "Interfaces to neighbouring designs" lists what this
spec assumes of them.

`crates/gsv` is the server side of everything a Switch 2 title's graphics driver talks to before
and while it draws: the graphics server ("gsv") and the sessions it hands out, the syncpoint
counters, the GPU's view of memory, GPU channels, the display (vi2) and the video codec sessions.

## Summary

- **One graphics server, six session families.** `am` hands the title a session to the gsv root
  (`ISelfController` 2000). The root hands out syncpoint, GPU, display, NVIDIA driver (nvsrv) and
  video sessions. All are TIPC. Each family is its own server in the service framework, so they
  answer in parallel on different cores.
- **Nothing in gsv blocks a core.** Work that waits on the GPU, a vsync or a decoded frame is a
  kernel event the guest waits on, or a deferred reply. Host threads (GPU, timer, video worker)
  report back through one completion queue, which the timer thread drains in a recorded order.
- **Syncpoints are 192 counters shared with the GPU,** a plain type in `crates/gpu` (decided by
  the owner, decision 3). The GPU increments them only once the host GPU has finished the work
  before them (owner's decision of 2026-11-12), and the video worker as it finishes frames. gsv
  allocates them, computes each submission's fence, mirrors them into the title's shared
  syncpoint page, and turns armed waits into kernel events.
- **GPU addresses equal CPU addresses.** gsv keeps the bookkeeping (registered memory, mappings,
  sparse regions) and writes a translation table the GPU reads (`GpuVm`, also in `crates/gpu`).
  A mapping that the guest's own memory backs resolves to the same CPU address. Unmaps, kind
  changes and destroys go to the GPU as ordered updates. GPU-only regions are backed lazily on
  the host by the GPU crate, outside the title's memory pools.
- **Submission is a hand-off.** nvsrv's channel ioctls validate a GPFIFO submission, compute its
  fence, and pass the entries to the GPU front thread. The reply does not wait for the GPU.
- **Present never blocks** (decided by the owner, decision 1: a buffer queue). vi2 229 queues
  the image with its fence and replies at once with the buffers still free. A 60 Hz vsync on the
  timer thread latches one ready frame per tick, hands it to the GPU, which orders it after
  earlier work and passes it to the render backend, and frees the buffer it replaces. That is
  what paces the title. The fence wait of the other reference is kept as a per-run fallback, as
  a deferred reply.
- **Video sessions exist from the start; decoding comes later** (decided by the owner, decision
  10). Sessions, image pools and syncpoint-backed frame handles are built so titles get past movie
  playback, with no decoding: movies show a frozen picture. FFmpeg comes later behind a Cargo
  feature that is off by default.

## How the protocol is known

Nothing here comes from Nintendo or NVIDIA documentation. Every fact comes from two
reverse-engineering efforts, both GPL-3.0-or-later, which disagree in places:

- **NX2:** `eden-nx2/src/core/hle/service/nx2_gsv.h @ 50fa86bb`, and its blocker log
  `eden-nx2/NX2_NOTES.md @ 50fa86bb`. "Row N" below is a row of that log, the same rows the
  scoping doc cites. NX2 was run against several titles and fixed one blocker at a time.
- **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/` `@ e5fb3b97`:
  `graphics_supervisor.cpp`, `graphics_supervisor_protocol.h`, `gpu_session.cpp`,
  `gpu_memory_registration.h`, `graphics_syncpoint_cache.h`, `native_image_layout.h`. Its
  comments cite addresses in the system modules themselves, so some of its readings come from
  the server, not only the client. It is a draft that ran fewer titles.

Each fact below carries one of four marks:

- **observed:** a reference answered this way and a title got further (a row says so).
- **read:** a reference read it from the SDK or the system module, without a row showing a title
  depending on it.
- **guess:** a reference marks it guessed, or neither says where it came from.
- **conflict:** the references disagree. The design picks one reading by the house rule of
  decision 8 (Nvidia's headers and readings that cite the system module for layouts and
  meanings; where one answer contains the other, the larger; otherwise the one a title ran on),
  says which, and lists the check that settles it ("What to verify before building"). The code
  marks each case and a test pins it.

Where a fact has a slice that ports it, the code repeats the citation next to the constant.

## The session tree

```text
am ISelfController 2000 ──► gsv root (TIPC)                         row 17
   ├─ 202 ─► syncpt session          slot events, armed waits        rows 19, 29, 43, 147
   ├─ 204    shared syncpoint page   two transfer memories, two events  rows 19, 28
   ├─ 497 ─► gpu session             notification events, address spaces  row 20
   │           └─ 21/22 ─► gpu worker   memory, maps, sparse regions     rows 27, 40, 42, 68, 85, 149
   ├─ 305 ─► vi2 session             displays, layers, present, vsync  rows 21, 48, 55, 82
   ├─ 1402 ─► nvsrv session          the nvdrv protocol: NvRm devices   rows 23–26, 45, 46, 149
   ├─ 902/906 ─► video context ─► decoder service                       rows 96, 105, 146, 148
   └─ 111–115, 314/315, 423–435, 911–991, 1510: native memory and images (answered by the root)
```

### The root

`am`'s `ISelfController` 2000 returns a raw session to the root (row 17, observed). `am` lives in
`crates/services` and may not depend on `gsv`, so it asks the service registry for a private
entry named `gsv` (not a name a title can open through `sm:`) to create the session. This uses
the registry the way the service framework intends ("Services talk to each other through the same
registry") and needs one addition there: entries that are reachable only from other services.

Root commands, by group:

| Commands | What | Mark |
|---|---|---|
| 4, 11, 101 | Bootstrap: 4 takes a copied process handle, the applet resource user id, a u64 flags word and four u64s; 11 binds the process; 101 completes. Flags `0x8000000000000005`–`7` with zero parameters are a local bootstrap; others ask for remote memory | read (PR `DecodeBootstrap`, `IsLocalBootstrap`); NX2 stubs all three |
| 126, 115 | Memory mode queries; reply 0 for "no remote memory" | read (PR) |
| 201, 202, 204 | Syncpoint context, syncpt session, shared page | 202/204 observed (rows 19, 28); 201 read (PR) |
| 301, 303, 305 | Display context; 305 takes a u64 (`0x3_0002_0000`, `0x3_0003_0000` or `0x3_0004_0000`) and returns the vi2 session | 305 observed (row 21); flags and 301/303 read (PR `IsSupportedDisplaySession`) |
| 401, 402, 434, 497 | GPU context; 434 registers an address space; 497 takes a u64 (4) and returns the GPU session | 497 observed (row 20); the rest read (PR) |
| 1402 | Returns the nvsrv session | observed (row 23) |
| 111, 112, 114, 115, 423 | Native memory: create (u8 mode, u64 size → u32 id), destroy, bind to process memory, bind to GPU-accessible memory | observed (rows 47, 48) |
| 429, 430, 435, 1510, 314, 315 | Native images and their size calculation; window alias image pools | observed (rows 33, 47) |
| 426, 427, 428 | Map, unmap, query a native image for the GPU | read (PR) |
| 901–906, 911, 912, 961, 991 | Video contexts and decoder image pools | observed (rows 96, 105, 146, 148) |
| 1001–1003 | **conflict:** a second decoder (NX2) or the video encoder (PR `InitializeVideoEncoder`). Both make 1002 return a session, which the codec start-up needs (row 105, observed) | Named as the encoder (PR cites the module); the reply is the same either way; verify |
| 1101/1102, 1111, 1121, 1201–1203, 1211, 1291, 1501/1502, 601/602 | Frame and image contexts, event supervisor, chat image pools, camera | read (PR); answered with PR's reply shapes, logged |

Every reply from these sessions is TIPC: a one-word result, then data. Eden's builder wrote the
two-word CMIF result there and shifted every data word (row 22, observed); the service framework
already gets this right. Returned sessions go at word 3 with the result at word 4 in NX2's
reading; PR moves them as move handles. Handle kind (copy or move) is a **conflict**; copy is
used, since titles ran on it (NX2), and a trace settles it.

**Bootstrap order.** PR enforces a phase machine (4, then 11, then 101, then the contexts) and
answers anything out of order with an error. NX2 accepts everything. Decision 7.

### Syncpoints and the syncpt session

A **syncpoint** is a 32-bit hardware counter that only goes up. Work that finishes increments it;
anyone who wants to know whether that work finished compares the counter with a threshold. A
**fence** is the pair (syncpoint id, value): "done when the counter reaches value".

- **Count:** 192 counters, the size of Eden's host1x syncpoint table, which both references keep
  (PR `GraphicsSyncpointCache::CounterCount`, NX2 `HostSyncpoints`). Guess: no reference says
  the console has 192.
- **Ids on the wire:** a `SyncCounterId` is the bitwise NOT of the syncpoint id (observed, NX2
  `QueryCounter`; read, PR `DecodeSyncpointWait`).
- **Comparison:** mode 0 compares 32-bit wrapping (`(current - threshold) as i32 >= 0`), mode 1
  compares as 64-bit (read, PR `IsSyncpointWaitSatisfied`; NX2 comment on TIPC 201).
- **64-bit values:** the shared page holds u64 values. A 32-bit counter is extended by counting
  wraps; an unallocated counter reads as all ones (read, PR `ExtendCounter`, `Unallocated`).
- **Allocation:** ids are allocated per channel and per video decoder, with Eden's reserved ids
  (`eden/src/core/hle/service/nvdrv/core/syncpoint_manager.cpp @ cf654052` reserves 26 and 27 for
  vblank). Ported as is; whether Switch 2 reserves the same ids is a guess.

The syncpt session (gsv 202):

| Command | Request → reply | Mark |
|---|---|---|
| 0 | u64 token → the same token. Newer SDKs abort without the echo | observed (row 43) |
| 1 | → u32 maximum event slots. PR answers 63, NX2 32; the SDK requires < 0x400 and passes `min(value, 63)` to 21 | **conflict** on the value; 63 chosen (PR) |
| 21 | u32 count → result. Slots at or above the count are cleared | read (PR `SetEventCount`) |
| 22 | u32 slot → an event handle for that slot | observed (row 29) |
| 103 | u32 SyncCounterId → u64 value, u32 0 | read (both) |
| 201 | u32 slot, u32 SyncCounterId, u64 threshold, u32 mode → result. Clears the slot's event, then signals it once the counter reaches the threshold | observed (rows 29, 147) |
| 202 | u32 slot → result. Cancels, and **always** signals the slot's event, because the SDK waits on it after every cancel | observed (row 147) |

A slot event is a kernel event gsv owns. Arming a wait registers (slot, fence) in gsv's waiter
table; it never blocks. When the counter reaches the threshold, the completion queue signals the
event (below, "Who signals what").

### The shared syncpoint page (gsv 204)

The title passes two transfer memories: 0x1000 bytes it writes, and 0x2000 bytes it maps
read-only (observed, rows 19 and 28). gsv writes a header into the read-only one: u32 version 1
at +0, u32 base id at +4, u32 count at +8 (the client requires `((count + 0x40) >> 6) - 1 < 0x40`),
then the offsets of five areas: the u64 counter values, the request summary word (writable page),
the acknowledge summary word (read-only page), the per-slot request bits (writable page) and the
per-slot acknowledge bits (read-only page). gsv returns two events: one the title signals to ask,
one gsv signals to answer (row 28: the first is attached writable).

The header is self-describing: NX2 and PR put the areas at different offsets and both worked for
their authors (NX2 `SyncptValuesOffset` and neighbours; PR `GraphicsSyncpointCache::Header`). So
the offsets are ours to choose; this design uses PR's (values at +0x40, acknowledges at +0x640 and
+0x648, requests at +0 and +8), which fit the 192 counters with room to spare. The count field is
also a **conflict**: NX2 writes the count, PR writes the count minus one.

**Protocol.** To ask for fresh values, the title flips request bits so they differ from the
acknowledge bits, and signals the request event. gsv writes the current values of the requested
counters, copies the request bits into the acknowledge bits (summary last, with a release
barrier), and signals the answer event (read, PR `RefreshSyncpointCache`). NX2 reads the same
bits as an armed wait the server answers when it likes; PR as a cache refresh answered promptly.
Both answer every outstanding request within a bounded time (NX2 polls every 500 µs, PR every
1 ms), and both work. This design answers a request **when the guest signals the request event,
when a requested counter changes, and at the latest 1 ms after the request** (decision 9). The
1 ms bound keeps a request for a counter nobody increments from hanging; the other two keep the
common case fast without a poller.

Answering when the guest signals needs one thing from the kernel: a hook on an event that gsv
created, called after the kernel lock is released (never under it), that queues a refresh. Until
the kernel has it, the 1 ms timer alone answers, as both references do.

### GPU memory: the gpu session and its worker

The GPU on a Switch 2 sees memory through GPU virtual addresses. Titles never map through the
old address-space ioctls; memory registered with worker 101 is used at **GPU address == CPU
address** (row 30, observed). GPU-only memory lives in sparse regions the server backs (rows 57,
80, 81, 85, observed).

The gpu session (gsv 497):

| Command | What | Mark |
|---|---|---|
| 11 | → u32 first, u32 last notification index. PR answers [0, 63], NX2 [0, 31] | **conflict**; 63 chosen |
| 12 | u32 index → that notification's event | observed (row 20) |
| 21, 22 | 21(index) asks for the worker session and signals the index's event; 22 returns the worker session | observed (row 20); names read (PR `RequestWorker`, `AcquireWorker`) |
| 33 | → u32 address space id (an nvhost-as-gpu address space) | read (PR `CreateAddressSpace`); NX2 answered 1 |
| 41, 42 | Create (u64 name, u64 flags 0, u64 label → u32 device id) and destroy a graphics device | read (PR) |
| 201, 202, 203 | Address-space partitions. **conflict:** NX2 has 201 = user partition, 202 = gsv partition, 203 = VA range; PR has 202 = gsv, 203 = user | Verify by trace; until then the NX2 reading, which a title ran on (row 44) |

The worker session, whose jobs complete by signalling a notification event the request names
(the "job slot"):

| Command | Request words (after the two header words) | Mark |
|---|---|---|
| 101 `CreateGpuAccessibleMemory` | u64 first, u64 last (inclusive), u8 flags, u8 device → u32 memory handle | observed (rows 27, 42); field layout read (PR `DecodeGpuMemoryRegistration`) |
| 102 `DestroyGpuAccessibleMemory` | u32 handle, u32 slot. Completes after the GPU is past all work queued before it | observed (row 68) |
| 211 `NormalMapGpuMemory4K` | address space (w2), first (w3–4), last (w5–6), memory handle (w7), offset into it (w8–9), permission, cache, kind, fallback kind, mode, device, sync (w10–16, one byte each), slot (w17) | observed (rows 40, 42); byte fields read (PR) |
| 212 `NormalModifyMemoryKind4K` | address space, first, last, wanted kind (w7), fallback (w8), sync (w9), slot (w10) → u32 kind applied | observed (row 149) |
| 213 `NormalUnmapGpuMemory4K` | address space, first, last, sync (w7), slot (w8). Completes after queued work | observed (rows 42, 68) |
| 214 `CreateSparseRegion` | address space (w2), first, last, page shift (w7: 12 or 16), allocation mode, device, sync, slot (w11) | observed (rows 57, 80, 85); fields read (PR `DecodeGpuSparseRegion`) |
| 215 `DestroySparseRegion` | as 213. Completes after queued work; the backing is freed only then | observed (row 85, NX2 `SubDestroySparseRegion`) |
| 216 `ExecuteSparseMapRequests` | a batch of up to 256 40-byte entries in a buffer (address, size, flags, kinds, memory handle, source offset) → u64 processed | read (PR only) |

Reply shapes for 211 and 213 are a **conflict**: PR replies with the selected kind (211) and the
bytes unmapped (213); NX2 replies with the result only. PR's are used (extra reply words are
harmless to a client that ignores them); verify.

#### The model

gsv keeps, per process:

- **Memory registrations** (101): a CPU range, checked against the kernel's address space: every
  page must have the GPU-accessible attribute that `svc 0x80` (`SetGpuMemoryAttribute`) sets
  (read, PR `RegisterMemory`). Registering takes the kernel's device-share lock on the range
  (the kernel spec keeps the device share count in every memory block from its first slice), so
  the CPU mapping under it cannot change while the GPU may use it. Destroying releases the lock
  only after the GPU is past the work queued before the destroy.
- **Address spaces** (gpu 33, and nvsrv's nvhost-as-gpu opens): one object reachable from both
  paths by id. NVN reserves 4 KiB pages from `0x8000000` to `0xC0_0000_0000` (row 25, observed).
- **Mappings** in each address space: GPU range → target, kind. The target is one of:
  - `Guest { cpu_va }`: memory the title owns. A 101 registration maps itself at its own address
    (GPU address == CPU address); a 211 maps part of a registration at a chosen GPU address,
    which can differ from the CPU address (for example inside a sparse region).
  - `GpuOnly { region, offset }`: inside a sparse region, where nothing of the title's is mapped.
  - `SyncpointShim`: the read-only page through which GPU reads return syncpoint values (row 46,
    nvhost-as-gpu ioctl 0xB reports where it is).
  - nothing: a hole.
- **Sparse regions** (214): a GPU range with no CPU mapping. Six 300 MB regions in one title, two
  of about 5.7 GiB in another (rows 57, 81).

Mapping a 101 registration at its own address on creation follows NX2 (row 30, observed: without
it the GPU ran nothing). PR maps only on an explicit 211. Decision 5.

#### How mappings reach the GPU

gsv owns the bookkeeping and validates every request against what it actually allocated
(handles, ranges, overlaps), as `CLAUDE.md` requires of guest-supplied ids. The GPU reads only a
**translation table** that gsv publishes. The table's type lives in `crates/gpu` (gsv depends on
gpu, not the other way; decided by the owner, decision 3). The GPU never sees gsv's bookkeeping.

- **A map** is published at once: no work queued before it can name the new range, because the
  title waits for the job's event before using it.
- **An unmap, a kind change, a destroy and a sparse-region destroy** are sent to the GPU as an
  ordered update (`GpuFront::update_vm`). The GPU applies it only after every piece of work
  queued before it has stopped reading through the old mapping, including the GPU thread that
  reads command buffers and the releases that wait for the host GPU, then reports the update done
  (`GpuCompletion::GpuOrdered`). Only then does gsv signal the job's event and release the
  device-share lock or the backing. NX2 crashed after a save when
  102 and 213 completed at once: stale GPU writes landed in memory the title had reused as heap
  (row 68, observed).
- The table is versioned. A GPU thread caches translations per thread and drops its cache when
  the version moves; a cache shared between threads crashed NX2 every few minutes (row 125,
  observed).

#### GPU-only memory

A sparse region has no memory of the title's behind it, but render targets live there (row 57).
NX2 backed each region with pages from the application pool and the title ran out of memory: the
pool still reported those pages free (row 80). This design backs GPU-only memory **lazily and on
the host**: the GPU crate (#12) owns the lazy store, because only GPU threads touch it (review
guide, M6). A render target in such a region stays a host-only texture in the GPU crate's
texture cache; other bytes get host storage the first time the GPU writes them. Nothing is
charged to the title's pools, and gsv keeps only the region extents and their accounting. Reads
of a part never written return zero. Decision 4.

Two things are unknown. Whether the console charges these regions to the title's *remote memory*
(`svcGetInfo` 31 and 32, the NPDM's remote memory size, and the bootstrap's remote mode) is not
known; NX2 answered 31 and 32 with 0 (row 16). And whether unmapped sparse pages read as zero is
a guess. Both are in "What to verify".

#### The semaphore mirror page

NX2 copies GPU releases that land in the second page of a two-page registration into the first
page, because NVN2 polls one and the GPU acquires the other (row 35, NX2 marks it "guess, works").
PR has no such copy. It is GPU behaviour, not a server command; gsv tells #12 which registrations
are two pages long (the only thing gsv knows), and #12 decides. Listed for verification.

### Channels and submission: nvsrv

nvsrv speaks the old nvdrv `INvDrvServices` protocol (Open, Ioctl, Close and their variants) over
a session gsv 1402 returns, opened without `Initialize` (row 23, observed). Switch 2's NvRm uses
these devices:

| Device | Used for | Source |
|---|---|---|
| `/dev/nvhost-ctrl-gpu` | A 0xD0-byte characteristics block: architecture 0x170 + implementation 0xE (chip 0x17E), flags with bits 8 and 22 (TSG, subcontexts), 26 (syncpoint shim), 34 (mapping modify: without it compressible textures get GPU address 0) and 35, 40–43; signed interface levels from +0x70; 64 subcontexts at +0xB4 | rows 24, 45, 46, 149 (observed); full layout read (PR `native_gpu_characteristics.h`) |
| `/dev/nvhost-as-gpu` | Address space allocation (AllocAsEx, flags 5, small pages), VA regions, shim info (0xB), ModifyKind (0xC), Remap (0x14) | rows 25, 46 (observed); 0xC, 0x14 read (PR `nvhost_as_gpu.cpp`) |
| `/dev/nvhost-tsg-gpu` | Timeslice groups and subcontexts | rows 26, 45 (observed) |
| `/dev/nvhost-gpu` | Channels: AllocGPFIFOEx2 (0x1A), AllocObjCtx (0x9; classes C797 3D, C7C0 compute, C7B5 copy, and class 0), SubmitGPFIFO (0x8, 0x1B), per-channel events (QueryEvent 7) | rows 26, 31, 34 (observed) |
| `/dev/nvmap` | Memory handles | Eden, unchanged |

These are ports of Eden's `src/core/hle/service/nvdrv/devices/` (`@ cf654052`) with the Switch 2
changes from both references. The address-space ioctls and the gpu worker act on the same
address-space objects. Unknown ioctls from a Switch 2 driver answer "not implemented" under the
strict policy and a zeroed success under auto-stub (the service framework's policy; NX2 echoed
the input instead, which the framework forbids after row 96).

**Submission.** SubmitGPFIFO carries GPFIFO entries (8 bytes each: a command buffer's GPU address
and length) and flags: bit 0 wait on the given fence first, bit 1 increment the channel's
syncpoint by 2 at the end, bit 4 skip the wait-for-idle, bit 8 increment by the given value (Eden
`nvhost_gpu.h` flags; NX2 `nvhost_gpu.cpp` `SubmitGPFIFOImpl`). nvsrv:

1. reads and bounds-checks the entries (count against the request's buffer, no wrap),
2. computes the increment (`2 if bit 1` + `value if bit 8`), raises the channel syncpoint's
   maximum by it, and sets the reply's fence to (channel syncpoint, new maximum),
3. hands the GPU front one `Submission` (`GpuFront::submit`): the entries, the optional fence to
   wait on, the number of increments to perform after them, and the wait-for-idle flag. NVN2 sets
   bit 8 and expects the increments to happen even though its commands contain none; the front
   appends them (row 32, observed), and the GPU performs them once the host GPU has finished the
   work before them,
4. replies at once. The GPU runs later; the title learns it finished through the fence.

Each channel's submission queue is bounded at 65,536 GPFIFO entries (decided by the owner,
decision 12; the same figure as gpu-protocol's "Back-pressure"). If the front's queue is full
(`Submitted::Full`), the request takes a deferred reply that gsv completes on
`GpuCompletion::GpuQueueRoom`. A handler never waits.

Channel creation (AllocGPFIFOEx2) allocates a syncpoint and asks the front to create a channel
in the right address space; binding an object class goes to the front unchanged. The engine work
behind both is #12's.

### Display: vi2

vi2 is the Switch 2 display service. The commands below are PR's table; PR names them from the
vi2 module (`vi2.autogen` addresses in its comments), so their meanings are read, not observed,
except where a row says so.

| Commands | What |
|---|---|
| 1, 3, 5 | Limits: 5 answers 7 (the SDK makes seven timestamp recorders per layer; observed, row 21) |
| 11, 13, 21 | A notification event and a 32-word bitmap of notifications; reading the bitmap clears it |
| 51, 31, 32 | Request groups (at most 8): 31 creates a group's event, 32 duplicates it |
| 101, 102 | Open and close a display by its 8-byte name (`"default"`) |
| 104–107, 354 | Display state: HDR status, maximum buffer resolution, other display records. 104 and 105 are called about 75 times a second and must not log per call (row 119) |
| 111, 112 | Open and close an endpoint on a display |
| 201, 202, 212 | Register a client (a u64 token and two notification bit indices: "retired" and "buffer available"), unregister it, read its availability mask (creates its layer on first use) |
| 228, 229 | Cancel or present one of the client's buffers |
| 251, 252 | Bind a client to a display |
| 261–272 | Register (odd) and unregister (even) a request group as a listener of one kind of display event; kind 0 is the vsync |
| 301, 351, 352 | Display flags, output encoding and output levels |

**Vsync events: conflict.** NX2 treats the handles from 31 and 32 as vsync events and signals
every one at 60 Hz; a title waited on the event from 32 until it had fired twice before
submitting any GPU work (row 82, observed). PR reads 31 and 32 as request-group events, signalled
at 60 Hz only after 261 registers the group for kind 0 (the vsync). Both explain row 82. This
design follows PR, since it describes the module, and keeps NX2's behaviour as a fallback: if a
group event handed out by 31 or 32 has no registration after the first frame, it is ticked
anyway, and the fallback is logged. The trace decides which one stays.

**Layers.** A registered client owns one layer on the display it is bound to. The first slices
support one display (`"default"`) and one visible layer per client; composition of several
layers and other displays is deferred.

#### Present (229) and pacing

Request (24 words, header 0xF5): u64 client token (w2–3), u8 buffer slot (w4), u32 native image
id (w5), u32 byte offset (w6, 0), u32 `(width - 1) | (height - 1) << 16` of the shown area (w7),
u32 layout (w8, 3), fence `SyncCounterId` (w22) and value (w24). Reply: result, u64 mask of the
client's slots now free, four zero words. Fifteen slots; a fresh client has all free (`0x7FFF`).
(Observed, row 48; word layout read, PR `QueueNativeImage`.)

The title paces itself on this reply. If the reply says every buffer is free, nothing stops the
CPU running a full ring of command memory ahead of the GPU (row 55, observed) or presenting at
160 frames a second (row 71, observed). There are two ways to pace it:

- **Buffer queue (PR, recommended).** 229 queues the image with its fence, clears the slot's bit
  and replies at once. The title cannot reuse that buffer until the slot comes back, which it
  learns from the "buffer available" notification. A slot comes back only after a later frame
  has replaced it on screen.
- **Fence wait (NX2).** 229 holds its reply until the previous frame's fence is reached and the
  next 60 Hz tick has passed, and reports every slot free. NX2 had to bound the wait at 4 s
  (row 59), saw 8–56 ms of every frame lost to it (row 108), and deadlocked when it waited for
  the current frame instead of the previous one at a scene change (row 108).

This design builds the buffer queue (decided by the owner, decision 1). The handler never waits:

1. 229 validates the client, slot and image, records `Queued { slot, image, fence }` in FIFO
   order, clears the slot's bit, and replies.
2. Each vsync tick (timer thread, 60 Hz): take the oldest queued frame whose fence is reached. Under
   the owner's ruling on GPU completion, "reached" means the host GPU has finished the work that
   made the frame. If there is one, it becomes the shown frame and is handed to the GPU as a
   `PresentRequest` (`GpuFront::present`; no fence, nothing waits). The frame it replaces is freed
   on the same tick: its slot's bit is set again, the client's "buffer available" bit is raised in
   the notification bitmap, and the notification event is signalled.
3. 228 (cancel) gives a slot back with no image.

Freeing on the tick, not on a report from the render backend, makes buffer return a pure function of
the 60 Hz clock (review guide, M3, option b; the owner's buffer-queue ruling: the tick "frees the
buffer it replaces"). It is safe only because the present and any later draw into the same image
reach the render backend in order and run in order on its one host queue (render-backend, "Submit").
That ordering is an inference the review guide flagged, so it is a **guess** until render-backend
slice R8's test pins it; if it fails, the release moves back to a render report.

**Fallback: the fence wait** (owner's decision of 2026-11-12). A per-run setting makes 229 hold
its reply until the previous frame's fence is reached and the next tick has passed, and report
every slot free, as NX2 does. It is written as a deferred reply completed by the timer thread, so
it never blocks a core. It exists in case a title turns out not to wait for free buffers
(verification 3).

At most one frame is latched per tick, so a title that waits for free buffers runs at 60 frames a
second at most, and at the rate the GPU sustains when that is lower. A frame that misses a tick
waits for the next one; lost time is not banked (NX2 `PaceToVblank`, which is the console's
behaviour as NX2 describes it).

**The vsync clock.** The 60 Hz tick is a timer-thread deadline computed from the `Env` clock
(decision 2), so a fixed clock (`--clock fixed`) makes it deterministic and headless runs need no
display. Every tick signals the vsync request groups and runs the latch above. The tick count is
recorded.

**What vi2 needs from the image.** A native image (gsv 429) carries a 16-word preference: word 2
is `(width - 1) | (height - 1) << 16`, word 4 the block height (log2 of GOB rows), word 5 the row
pitch in 64-byte GOBs, word 6 the format (1 RGBA8, 4 RGB10A2) (observed, row 48; read, PR
`native_image_layout.h`). Its memory is bound by 114 or 423, so its GPU address is known. vi2
passes the GPU address, size, pitch, block height, format and the shown area to the GPU front;
the GPU crate detiles (the Switch 2 GOB swizzle, row 69) with the layout crate
`starquake-layout`, and the render backend receives linear data or the GPU's own render target
(review guide, M8).

### Native memory and images

Native memory (111) is an id with a size; 114 or 423 binds it to an address. Native images (429)
are a preference plus a memory id. Their size, for 435 and 1510, is
`pitch_gobs × 64 × align(height, 1 << block_height) × count`, aligned to one page (read, PR
`CalculateNativeImageRequirements`, "run against the real module"; NX2's alignment differs:
**conflict**, PR's chosen). gsv keeps this formula because it reproduces the system module's
answer (a protocol fact); a test checks it against `starquake-layout`'s size for the same
description, which needs the layout crate as a dev-dependency of gsv (review guide, M8). 1510
returning size 0 made the window fail to initialise (row 33, observed). These are small tables in gsv's root, shared by vi2 (presents) and video (pools).

### Video codec sessions

What exists, all through gsv:

- **Contexts.** 901 initialises, 902 (no arguments) or 906 (u8, u64 library version, newer SDKs)
  returns a context session whose TIPC 1 returns a decoder service session (rows 105, 146,
  observed). 903 finalises.
- **Image pools.** 991 sizes a pool (from a decode preference: an output layout code at word 1,
  the size at word 2; codes 1 and 4 are NV12), 911 creates `count` images that share one memory,
  961 returns a 128-byte CPU image descriptor (kind 8, 9 or 10 for pitch-linear RGBA, BGRA, NV12;
  plane offset and pitch) (rows 96, 148, observed). An auto-stubbed 991 echoed the request and
  aborted a title (row 96): these three are answered from the first slice.
- **The decoder service.** 200 initialises with a work-buffer transfer memory; 101 names the codec
  (`"video/vp9"`) and the size; 3 returns a "ready for input" event; 109 submits one compressed
  frame; 311 and 314 register and give back output images, each answered with a native sync
  handle {SyncCounterId, mode, u64 value} on a syncpoint the service allocates and increments per
  finished frame; 213 returns the next finished frame; 6, 207, 8, 115 start, stop, flush,
  finalise (row 148, observed; NX2 `INx2VideoSession` comment block).
- **Encoder** (1001–1003 in PR's reading), camera (601, 602): sessions with PR's reply shapes,
  everything else logged.

What is deferred: the decoding. The first slices answer every command and complete each submitted
frame with no picture (the image keeps its contents), incrementing the decoder's syncpoint in
order, so a title gets through its movies with a frozen picture. That is NX2's state before
row 148, which was enough to get past the aborts. Real decoding runs on a worker thread per
decoder service, writes into guest memory through `GuestMemory`, and reports each finished frame
through the completion queue. It comes later: FFmpeg behind a Cargo feature that is off by
default, covering VP9, AV1 and H.264, with each format moving to a pure-Rust decoder when a good
enough one exists (decided by the owner, decision 10). The decoder sits behind the session
protocol, so it can be swapped without changing the sessions. A title that waits for a movie to
finish may sit on a frozen frame; that is found when a real title hits it.

## Waiting without blocking a core

Every wait in this design is one of two kinds:

| Wait | Mechanism | Completed by |
|---|---|---|
| syncpt 201 armed wait, slot events | Kernel event the guest waits on | Completion queue, when the counter reaches the threshold |
| Shared-page request (204) | Kernel event the guest waits on | Completion queue: request-event hook, counter change, or the 1 ms deadline |
| gpu worker jobs (101–216) | Notification event named by the request | At once (maps, creates), or when the GPU reports an ordered update done (unmaps, destroys) |
| vi2 vsync request groups | Kernel event | Vsync tick |
| vi2 buffer available | Notification bit + event | The vsync tick that latches the frame replacing it |
| vi2 229 in the fence-wait fallback | Deferred reply | Vsync tick, once the previous frame's fence is reached |
| SubmitGPFIFO with a full GPU queue (65,536 entries, decision 12) | Deferred reply | GPU front, when it has room |
| Video ready-for-input, frame done | Kernel event; syncpoint fence | Video worker |

### Who signals what: the completion queue

Host threads never touch gsv's tables or the kernel directly. They push a `Completion` onto one
multi-producer queue owned by gsv and wake the timer thread. The GPU's part of it is a type in
`crates/gpu` (`GpuCompletion`, pushed through the `Completions` trait gsv implements), because the
GPU cannot name a gsv type:

```rust
enum Completion {
    Gpu(GpuCompletion),                                  // see "What this spec offers ... GPU front/back"
    VideoFrame { decoder: DecoderId, frame: FrameInfo }, // video worker
    GuestSignalled { event: ObjectId },                  // kernel hook on gsv's own events
}
```

The draft's `PresentReleased` completion is gone: no host thread reports a presented buffer free
(the vsync tick frees it).

The timer thread drains the queue (and runs gsv's deadlines: vsync, the 1 ms shared-page bound):
it updates the waiter tables, writes the shared page, signals kernel events and completes
deferred replies. Draining on one thread gives one order, which is the order recorded for replay
(observability rule 3). `GpuCompletion::SyncpointChanged` is pushed only for counters someone
waits on: gsv keeps an atomic bitmap of watched ids that the incrementing thread checks first, so
the GPU pays one load per increment otherwise.

Lock order: gsv's shared state lock, then the kernel lock. Nothing calls into gsv while holding
the kernel lock or a GPU lock; the kernel's event hook queues a completion after releasing its
lock.

### Concurrency inside gsv

The service framework runs one request at a time per server. gsv registers one server per
family (root, syncpt, gpu, vi2, nvsrv, video; decision 6), so a present and a submission on two
cores do not wait for each other. State that both request handlers and the timer thread touch
(waiter tables, vi2 queues, the GPU memory bookkeeping, the native object tables) lives in one
`Arc<Mutex<GsvShared>>` with short critical sections that never touch guest memory under the
lock for longer than a copy.

## Interfaces to neighbouring designs

The four graphics drafts were amended together on 2026-11-12 so that every shared type and call
below has the same name in each. Field names may still change in implementation.

### What this spec offers and assumes of the GPU front/back (#12)

Types gsv needs to share with the GPU live in `crates/gpu`, because gsv depends on gpu (decided
by the owner, decision 3). The four graphics drafts were amended together on 2026-11-12, and
this block is the same in gpu-protocol's "Graphics server sessions (#11)":

```rust
pub struct SyncpointId(u8);                 // < SYNCPOINT_COUNT (192)
pub struct Fence { pub id: SyncpointId, pub value: u32 }
pub struct Syncpoints { /* [AtomicU32; 192], max values, watched bitmap */ }
impl Syncpoints {
    pub fn read(&self, id: SyncpointId) -> u32;
    pub fn increment(&self, id: SyncpointId) -> u32;   // pushes SyncpointChanged if watched
    pub fn reached(&self, fence: Fence) -> bool;       // 32-bit wrapping comparison
}

pub struct GpuVm { /* versioned translation table: GPU range -> Target, kind */ }
pub enum Target { Guest { cpu_va: u64 }, GpuOnly { region: RegionId, offset: u64 }, SyncpointShim }
pub enum VmUpdate { /* unmap, kind change, memory destroy, sparse-region destroy; each with its range */ }

pub struct ChannelConfig { pub space: AsId, pub syncpoint: SyncpointId }
pub struct Submission { pub entries: Vec<GpfifoEntry>, pub wait: Option<Fence>, pub increments: u32, pub wfi: bool }
pub enum Submitted { Queued, Full }         // Full: the channel's queue holds 65,536 entries
pub struct PresentRequest {
    pub image: ImageDesc,        // GPU address, width, height, pitch, block height, format
    pub shown: Rect,             // the area word 7 of 229 names
    pub layer: LayerId,
    pub display: DisplayId,
}

pub trait GpuFront: Send + Sync {                       // implemented by `Gpu` in crates/gpu
    fn create_address_space(&self, space: AsId, vm: Arc<GpuVm>);
    fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>); // ordered
    fn create_channel(&self, config: ChannelConfig) -> Result<ChannelId, GpuError>;
    fn bind_class(&self, channel: ChannelId, class: u32) -> Result<(), GpuError>;
    fn submit(&self, channel: ChannelId, submission: Submission) -> Submitted;
    fn barrier(&self, done: Ticket);          // done once all work queued before it is finished
    fn present(&self, request: PresentRequest);   // no fence: gsv latched it after its fence was reached
    fn channel_error(&self, channel: ChannelId) -> Option<ChannelError>;
    fn destroy_channel(&self, channel: ChannelId);
}

pub enum GpuCompletion {
    SyncpointChanged { id: SyncpointId, value: u32 },   // back thread (after host completion), video worker
    GpuOrdered { ticket: Ticket },                       // an ordered update or barrier is done
    GpuQueueRoom { channel: ChannelId },                 // a full queue has room again
}
pub trait Completions: Send + Sync { fn push(&self, completion: GpuCompletion); } // implemented by gsv
```

This draft first placed `PresentRequest` among the render backend's types and the whole
`Completion` enum in gsv; both moved to `crates/gpu`, because the GPU consumes the one and
produces the other and cannot depend on gsv. `present` lost its release ticket (see "Present
(229) and pacing").

What gsv assumes:

- **Order.** Work, increments, ordered updates and presents take effect in the order `submit`,
  `update_vm`, `barrier` and `present` were called, across all channels of one process, as Eden's
  single GPU thread does. Which of the front or back thread performs each is #12's choice.
  "Finished" for an ordered update means no work queued before it can still use the old mapping,
  including the GPU thread that reads command buffers through the same table (owner's decision
  of 2026-11-12). gpu-protocol lists two cases this leaves open (a parked channel holding an
  update; later work seeing the old mapping until it applies).
- **Syncpoint increments** happen once the host GPU has finished the work before them, not when
  the work is recorded (owner's decision of 2026-11-12; this draft's earlier record-time
  assumption is dropped). Signals with no host work behind them happen at once. Readbacks of
  GPU-written memory by the CPU are #12's problem (rows 115, 131).
- **Fence waits** in a submission (flag bit 0) are waits on `Syncpoints` inside the GPU; gsv
  never waits for them.
- **Translation.** The GPU resolves addresses only through `GpuVm`, caches per thread, and checks
  the version. `Target::Guest` addresses are read through `GuestMemory` at the CPU address.
  `SyncpointShim` reads return `Syncpoints::read`. `GpuOnly` storage and its lazy allocation are
  #12's.
- **No callbacks into gsv except `Completions::push`,** which never blocks.

What gsv gives #12: the GPU profile from the bootstrap (flags 5 → 0x807, 6 and 7 → 0x808; read,
PR `NativeGpuProfileForBootstrap`), which two-page registrations exist (the semaphore mirror
question), and which classes a channel binds.

### What this spec assumes of the shader pipeline (#13)

Nothing at runtime. gsv passes the GPU profile above to #12, which may hand it on. Row 65 says
binding the console's blit constant bank 24 is "the system's job"; no gsv command is known to do
it. gpu-protocol and shader-pipeline agree that #12 supplies it; the fact itself is open
(gpu-protocol verification 6). Unmaps and destroys reach the shader cache only through #12, which
calls its `invalidate` when it applies the ordered update.

### What this spec offers and assumes of the render backend (#14)

Nothing directly: presents reach the render backend only through the GPU
(`GpuFront::present`), which orders each after the work that produced the image and records it
as the render backend's `Command::Present`. The dependency direction is gsv → gpu → render.

- The render thread owns the window or swapchain, scales the shown area to it, and in a headless
  run writes `frames/` as the observability spec says.
- gsv frees a presented buffer on the tick that latches the next frame, not on a report from the
  render backend. This relies on the render backend running the present and any later draw into
  the same image in order on one host queue (render-backend, "Submit"; a **guess** until its
  slice R8 pins it).
- The render backend's `PresentFeedback` is statistics only; gsv never lets it change
  guest-visible behaviour.
- Vsync comes from gsv's timer, not from the host display (decision 2). A later option is to let
  the host display drive it; `PresentRequest` and the latch do not change.

### What this spec needs from the kernel and the service framework

- Events gsv creates, signalled and cleared from the timer thread; the event hook above.
- Transfer memory (204, video 200) and handle copy and move; a kernel spec "later slice".
- Address-space queries: the GPU-accessible attribute (`svc 0x80`) over a range, and the
  device-share lock and unlock (kernel spec rule 4).
- Deferred replies completed from the timer thread.
- A private registry entry for `am`'s `ISelfController` 2000.

## What is recorded

The service framework already records one IPC event per request. gsv adds `ServiceNote` events
for every input replay cannot reproduce, all from the timer thread as it drains the queue:

| Note | Fields |
|---|---|
| `SyncpointObserved` | id, value (each value gsv acts on) |
| `SyncptPageAnswered` | request bits answered, values written |
| `GpuOrderedDone` | ticket |
| `VsyncTick` | tick number |
| `FrameLatched`, `BufferReleased` | client, slot, image (both written on the vsync tick that latches the frame) |
| `VideoFrameDone` | decoder, image, pts |
| `QueueRoom` | channel |

Submitted GPFIFO entries are part of the IPC payload, recorded in full mode like any request
data. GPU method streams are not events (observability spec); they are #12's capture format.

Replaying the kernel and services from a log replays gsv with no GPU: a fake `GpuFront` takes
submissions, and the recorded notes are fed back in their order. Because a buffer is freed by the
tick and not by a host report, buffer return replays from the tick count alone.

## What is snapshotted

`GsvShared` and each session are plain serialisable data:

- syncpoint allocation, values and maxima, the watched bitmap and every armed wait;
- the shared page's location and the last request bits answered;
- memory registrations, address spaces, mappings, sparse regions and their GPU-only extents;
- native memories and images; vi2 displays, clients, request groups, queued and shown frames,
  availability masks, notification bitmaps, the vsync tick and next deadline;
- channels (id, syncpoint, address space, bound classes) as gsv knows them; their engine state is
  #12's;
- outstanding tickets and deferred replies, as ids.

Two things are not in gsv's snapshot: the contents of GPU-only memory (on the host GPU; a GPU
snapshot, #12/#14, must read it back) and a video decoder's internal state (a snapshot taken
mid-movie resumes with the decoder reset, and the title sees one frame without a picture).

## Switch 1 paths

- **nvdrv ioctls: needed.** nvsrv is the nvdrv protocol (row 23), and Switch 2's NvRm uses the
  devices above. They are ported once and served behind gsv 1402.
- **The `nvdrv` named services and the old `vi` services (`vi:u`, `vi:m`, the binder and
  nvnflinger): not needed for Switch 2 titles.** Neither reference's blocker log shows a Switch 2
  title opening them; NX2 reaches gameplay without them, and PR uses Eden's old vi only inside
  its vi2 implementation as a compositor. A trace of `sm:` requests (the service framework logs
  every name) would show it if a title did. They are deferred: registering `nvdrv` later reuses
  the same devices; old vi would need its own service, written on vi2's display and layer model.

## Port slices

Each slice is one pull request with its own tests, and records and serialises from the start.
All slices test offline: no title, no GPU, no hypervisor. The tools are the IPC request builder
(requests laid out by hand from the word layouts cited above, never by a builder written in the
same slice), the kernel's fake for events and sessions, the memory crate's in-memory
`GuestMemory`, a recording fake `GpuFront`, and a fixed `Env` clock.

Paths in the table are short forms. "Eden" is `eden/src/core/hle/service/` `@ cf654052`. "PR" is
`eden-pr4588/src/core/hle/service/am/service/` `@ e5fb3b97`, except PR files under `nvdrv`, which
are in `eden-pr4588/src/core/hle/service/nvdrv/devices/`. "NX2" names classes in
`eden-nx2/src/core/hle/service/nx2_gsv.h` `@ 50fa86bb`; NX2 files under `nvdrv` or `am` are in
`eden-nx2/src/core/hle/service/`.

| # | Slice | Builds | Depends on | Ports from (`@` commits as above) | Offline test |
|---|---|---|---|---|---|
| 1 | Protocol facts and the root | The command table with marks and sources; the root's bootstrap (4, 11, 101, 126, 115) and every root command left as a logged stub with its known reply shape; the private registry entry; `am` 2000 | Services slices 1–3 | PR `graphics_supervisor_protocol.h`, `graphics_supervisor.cpp` (root); NX2 `INx2GraphicsSystemSession`; NX2 `am/service/self_controller.cpp`; PR `self_controller.cpp` | Each root command's reply bytes; bootstrap out of order is logged, not refused (decision 7) |
| 2 | Syncpoints | `Syncpoints` in `crates/gpu`: counters, maxima, wrapping and 64-bit comparison, 64-bit extension, allocation with reserved ids, watched bitmap, the `GpuCompletion` and `Completions` types | None | Eden `nvdrv/core/syncpoint_manager.cpp`; PR `graphics_syncpoint_cache.h`, `IsSyncpointWaitSatisfied` | Comparisons at the wrap edge on both sides; extension across wraps; allocation exhausts at exactly the count |
| 3 | Completion queue and timer hooks | `Completion` (wrapping `GpuCompletion`), the queue and its `Completions` implementation, the drain on the timer thread, the watched-id fast path, lock order | 2; kernel slice 10 (timers), with a fake until then | New | Scripted completions drain in push order; an unwatched increment pushes nothing |
| 4 | syncpt session | 0, 1, 21, 22, 103, 201, 202; waiter table | 1, 2, 3 | NX2 `INx2SyncptSession`; PR `ISyncpointSupervisor` | Wait fires exactly at the threshold, not one before; cancel always signals; arm clears a signalled slot (row 147) |
| 5 | Shared syncpoint page | 204: header, request and acknowledge protocol, the three answer triggers | 1, 3, 4; kernel transfer memory | PR `BindSyncpointMemory`, `RefreshSyncpointCache`; NX2 `AttachSyncptMemory` | Golden bytes of the header; a request is answered on signal, on change and at exactly the 1 ms bound |
| 6 | GPU VM table | `GpuVm` in `crates/gpu`: ranges, targets, kinds, versioning, overlap rules | None | New; NX2 `Nx2GpuMemoryRegistry` for behaviour | Model test against a page-by-page map under random map, unmap and modify |
| 7 | gpu session and worker | 497 commands; 101, 102, 211–216; registrations with the attribute check and device lock; sparse regions; ordered completion of unmaps and destroys | 1, 3, 6; kernel address space (attribute, device lock) | PR `gpu_session.cpp`, `gpu_memory_registration.h`; NX2 `INx2GpuSession` | Every request's bounds checks at the edge; an unmap's event fires only after the fake front reports the barrier; a destroy releases the lock only then |
| 8 | Native memory and images | 111, 112, 114, 115, 423, 429, 430, 435, 1510, 314, 315; layout sizes | 1, 7 (423); gpu-protocol G5 for the cross-check | PR `native_image_layout.h`, root native-memory handlers; NX2 `Nx2CalculateImageLayout` | Sizes against hand-computed values at block and pitch limits; invalid preferences are errors; the size formula agrees with `starquake-layout` for the same description |
| 9 | nvsrv framing and nvmap | The nvdrv protocol (Open, Ioctl variants, Close, events), ioctl number decoding, nvmap | 1 | Eden `nvdrv/nvdrv_interface.cpp`, `devices/nvmap.cpp`, `ioctl_serialization.h` | Ioctl byte fixtures; unknown ioctl under both policies |
| 10 | nvsrv GPU devices | ctrl-gpu characteristics, as-gpu (shared with slice 7's address spaces), tsg-gpu | 6, 7, 9 | Eden `devices/nvhost_ctrl_gpu.cpp`, `nvhost_as_gpu.cpp`; PR `native_gpu_characteristics.h`, `gpu_address_space_layout.h`, `nvhost_as_gpu.cpp`, `nvhost_tsg_gpu.cpp` | Golden bytes of the 0xD0 block, every listed bit; AllocAsEx ranges |
| 11 | Channels and submission | nvhost-gpu: AllocGPFIFOEx2, AllocObjCtx, SubmitGPFIFO, channel events; deferred reply on a full queue (decision 12) | 2, 3, 10; #12's `GpuFront` (a fake here) | Eden `devices/nvhost_gpu.cpp`; NX2 `nvhost_gpu.cpp` | Fence arithmetic for every flag combination; entries reach the fake in order; a queue at exactly 65,536 entries defers the next submit, one below does not, and the deferred reply completes on room |
| 12 | Vsync clock | The 60 Hz deadline from `Env`, no banking of lost time, request-group ticks | 3 | NX2 `PaceToVblank`; PR `StartGroupTick` | Fixed clock: ticks at exact multiples; a late tick does not double up |
| 13 | vi2 session | Displays, clients, layers, request groups and monitors, notifications, display state | 1, 3, 12 | PR `IVi2Session`; NX2 `INx2Vi2Session` | Each command's reply bytes; monitor kind 0 ticks; the fallback ticks an unregistered group and logs |
| 14 | Present and pacing | 228, 229, the FIFO latch, release on the latching tick, availability, the fence-wait fallback as a deferred reply | 2, 8, 13; #12's `GpuFront` (a fake here) | PR `QueueNativeImage`, `RefreshReleasedBuffers`; NX2 `PaceToVblank` for the fallback | Scripted presents and ticks: one latch per tick, an unreached fence skips a tick, a slot returns exactly on the tick that latches the frame replacing it and not before; the fallback's reply completes on the first tick after the previous frame's fence |
| 15 | Video sessions without decoding | 901–906, 911, 912, 961, 991, 426–428, the decoder service with a null decoder, 1001–1003 and 601/602 shapes | 2, 3, 8 | NX2 `INx2VideoSession`, `CalculateDecoderImagePoolBuffer`; PR `native_video_layout.h` | Pool sizes; sync handles count frames in order; flush keeps images already handed out |
| 16 | Wiring | `core` builds `Syncpoints`, the queue, gsv's servers and #12's front; recorded notes replay with a fake front | All above; #12 | New | A recorded gsv session replays to the same state |

**What can run in parallel.** Slices 2 and 6 are pure data structures (in `crates/gpu`, where
gpu-protocol's slice G22 builds the production memory access on them) and can start the moment
this spec is approved, with slice 1. After slices 1–3: slices 4, 7, 9 and 12 in parallel. Then 5
(after 4), 8 and 10 (after 7), 13 (after 12). Then 11 (after 10), 14 (after 8 and 13) and 15
(after 8), in parallel. Slice 16 is last and is single-agent work. Slices 2, 6, 11 and 14 touch
the seam with #12 and #14; those specs were approved with this one, so they code against the
shared interface above.

## What to verify before building

Traces from a title (local only, never committed) and reading the SDK settle these. Each could
change a decision.

1. **Which reference wins where they disagree:** gpu 201–203, gpu 11's range, syncpt 1's value,
   211 and 213 reply shapes, 1001–1003, copy or move handles. Until a trace, each follows the
   house rule of decision 8. A trace of the replies the title reads (its next request shows what
   it took from the last reply) settles each.
2. **vi2 31/32:** does the title register request groups with 261 before waiting on them? (vsync
   fallback, slice 13)
3. **Does the title wait for the "buffer available" notification before reusing a buffer?** If
   not, the owner's decision 1 falls back to NX2's fence wait for that run (the per-run setting).
   Count presents per vsync with the buffer queue.
4. **Does GPU work read a 101 registration at its own address without a 211 there?** Log the
   first GPU access per registration. (decision 5)
5. **Remote memory:** bootstrap flags, `svcGetInfo` 31 and 32, the NPDM's remote memory size: are
   sparse regions charged to the title's remote memory on a console? (decision 4)
6. **How much GPU-only memory is touched,** and whether anything reads a sparse page never
   written. This sizes lazy backing and checks the "reads zero" guess.
7. **The shared page's request rate** and how long requests stay unanswered without the 1 ms
   bound. Decides whether the bound matters.
8. **The semaphore mirror page (row 35):** with a correct GPU, is NVN2's wait satisfied without
   copying releases? (#12's question, gsv provides the two-page list)
9. **nvsrv framing:** CMIF or TIPC. NX2 serves it with Eden's CMIF nvdrv; nothing shows the wire.
10. **Syncpoint count and reserved ids** on Switch 2 (192 and 26/27 are Eden's Switch 1 values).
11. **Who binds constant bank 24** (row 65).
12. **Freeing a buffer on the latching tick is safe** (render-backend slice R8's ordering test):
    if a draw into a just-freed buffer can change the frame still being shown, the release moves
    back to a report from the render backend, through the GPU.

## Decisions for the owner

Every item carries its outcome; the original text stays below it, for the record. The rulings
are in `docs/decision-log.md` (2026-11-12).

1. **Decided by the owner, 2026-11-12:** the buffer queue, with the fence wait as a per-run
   fallback written as a deferred reply.

   **How presents pace the title.**
   - *Buffer queue:* the present reply comes back at once and says which buffers are still busy;
     a buffer comes back after a later frame has replaced it on screen. This is how one reference
     reads the console. Nothing waits inside the emulator, and frames can overlap with GPU work.
   - *Fence wait:* the reply is held until the previous frame is done and the next vsync passes,
     and says every buffer is free. The other reference ran several titles this way, but needed a
     4 s safety timeout, lost up to half of each frame to the wait, and deadlocked when tuned.
   - Recommendation: **the buffer queue**. It never blocks and does not trade speed for safety.
     The fence wait is kept as a per-run fallback in case a title turns out not to wait for free
     buffers (verification 3), written as a deferred reply so that it does not block a core
     either.
2. **Implied by decision 1, 2026-11-12:** our own timer; the buffer-queue ruling names "the
   graphics server's emulated 60 Hz clock". The same question is render-backend decision 8.

   **Where the 60 Hz vsync comes from.**
   - *Our own timer,* from the injectable clock: deterministic, replayable, works headless.
   - *The host display's refresh:* smoother on a 60 Hz monitor, but differs per machine, needs a
     window, and can't be replayed.
   - Recommendation: **our own timer**. The host display can drive it later without changing the
     design.
3. **Decided by the owner, 2026-11-12:** in `crates/gpu`, as plain data types; gsv decides and
   the GPU applies, with ordered unmaps reported only when no earlier GPU work can still use the
   old mapping.

   **Where the syncpoint counters and the GPU's translation table live.**
   - *In `crates/gpu`,* as plain types gsv creates and fills. The dependency direction
     (gsv → gpu) already allows it.
   - *In a new small crate* both depend on: cleaner ownership, one more crate.
   - *In gsv, behind traits the GPU calls:* the GPU would pay a virtual call per translation and
     per increment.
   - Recommendation: **in `crates/gpu`**. It needs no new crate and keeps the hot path direct.
4. **Decided by the owner, 2026-11-12:** lazily on the host, not charged to the title's pools,
   revisited if a trace shows the console charges it to remote memory. The GPU crate builds the
   store (gpu-protocol slice G22); gsv keeps extents and accounting.

   **How GPU-only memory is backed.**
   - *Lazily on the host:* storage is created when the GPU first writes a part of a region, and
     the title's memory pools are not charged. Huge reservations cost nothing until used.
   - *Up front from the title's pools* (one reference's first attempt): simple, but titles ran out
     of memory because the pools still reported the memory free.
   - *Not backed at all:* render targets there would be lost; frames went black this way.
   - Recommendation: **lazily on the host**, revisited if the console turns out to charge these
     regions to the title's remote memory (verification 5).
5. **Decided by the owner, 2026-11-12:** yes, as recommended.

   **Whether registering memory for the GPU also maps it at its own address.**
   - *Yes* (one reference): it observed nothing running without it, though its explicit map
     command was a stub at the time.
   - *Only when the title asks* (the other): closer to a literal reading of the commands.
   - Recommendation: **yes**, because GPU address == CPU address makes the extra mapping harmless
     and a missing one fatal; a trace can remove it later (verification 4).
6. **Decided by the owner, 2026-11-12:** one server per family, as recommended.

   **One server for everything, or one per session family.**
   - *One server:* a present waits behind a submission on another core, and so on.
   - *One per family* (root, syncpt, gpu, vi2, nvsrv, video), sharing a state object behind a
     lock: they run in parallel. Probably closer to the console, where vi2 is a separate module.
   - Recommendation: **one per family**.
7. **Decided by the owner, 2026-11-12:** log first, enforce once a trace confirms the order, as
   recommended.

   **Whether the bootstrap order is enforced.**
   - *Enforced:* one reference refuses commands out of order. Its error codes are its own guesses.
   - *Logged:* accept and log, as the kernel does with capabilities at first.
   - Recommendation: **log first, enforce once a trace confirms the order**.
8. **Decided by the owner, 2026-11-12:** one house rule for this spec and gpu-protocol: Nvidia's
   headers and readings that cite the system module for layouts and meanings; where one answer
   contains the other, the larger; otherwise the one a title ran on. Each case is marked in the
   code and pinned by a test.

   **Which reference wins when they disagree, until a trace settles it.**
   - Recommendation: **where one answer contains the other (a wider range, extra reply words a
     client can ignore), the larger one; otherwise the one a title ran on; when both ran or
     neither did, the one that cites the system module**. Each choice is marked "conflict" in the
     code and listed in verification 1. The alternative, deciding case by case without a rule,
     makes reviews argue each one again.
9. **Decided by the owner, 2026-11-12:** on demand, at most 1 ms stale; the 1 ms bound alone until
   the kernel has the event hook.

   **How the shared syncpoint page is answered.**
   - *Polled* every 0.5–1 ms, as both references do: simple, adds up to a millisecond to waits,
     and costs a timer wake-up 1,000–2,000 times a second.
   - *On demand:* when the title signals, when a watched counter changes, and at the latest after
     1 ms. Faster, and idle when nothing is asked.
   - Recommendation: **on demand with the 1 ms bound**; until the kernel offers the event hook it
     is the 1 ms bound alone.
10. **Decided by the owner, 2026-11-12:** no decoding in the first slices, then FFmpeg behind an
    off-by-default Cargo feature, each format moving to a pure-Rust decoder when a good enough one
    exists.

    **How video is decoded, when it is.**
    - *FFmpeg,* behind a Cargo feature that is off by default: covers VP9, AV1 and H.264 today, but
      is a C library, which the project keeps out of the default build.
    - *Pure-Rust decoders:* fit the project's rules; AV1 has a Rust port of a mature decoder, VP9
      and H.264 do not have a mature one (unverified).
    - *No decoding:* movies play with a frozen picture; titles carry on.
    - Recommendation: **no decoding in the first slices, then FFmpeg behind an off-by-default
      feature**, replaced by Rust decoders where one is good enough.
11. **Decided by the owner, 2026-11-12:** only the ioctl subset, as recommended.

    **Which Switch 1 paths to build.**
    - *Only the nvdrv ioctls Switch 2's driver uses, behind gsv:* what the evidence shows titles
      need.
    - *Also the `nvdrv` and old `vi` named services:* useful only for Switch 1 titles.
    - Recommendation: **only the ioctl subset**; the named services are deferred (section
      "Switch 1 paths").
12. **Decided by the owner, 2026-11-12:** bounded at 65,536 entries; a full queue makes the submit
    wait as a deferred reply.

    **What happens when the GPU's queue is full.**
    - *A bounded queue; the submission waits as a deferred reply* until there is room, like a
      console channel whose ring is full.
    - *An unbounded queue:* simpler, but a title that runs far ahead grows host memory without
      limit.
    - Recommendation: **bounded, with a deferred reply**.

## Deferred, not ruled out

| Deferred | How it could be added |
|---|---|
| Real video decoding (decision 10: FFmpeg behind an off-by-default feature, then pure-Rust decoders per format) and the encoder | A worker per decoder service, reporting through the completion queue; the session protocol is already in place |
| Several displays and layers, overlay and player displays, HDR, output encodings beyond the default | vi2 keeps displays and layers as tables from the start; composition is a render-backend change |
| Docked and handheld output sizes | Display state replies (106 and others) read from a display description instead of constants; `am` reports the mode |
| Remote-memory bootstrap and charging GPU-only memory to it | The bootstrap already records the mode; accounting moves from "host only" to the kernel's pools |
| Host display-driven vsync, variable refresh, unlocked frame rates | Replace the timer deadline with a host signal behind the same tick |
| The `nvdrv` named services and the old `vi` services for Switch 1 titles | Registry entries over the same nvdrv devices; old vi as its own service on vi2's display model |
| A real graphics server module run as a guest process | The service registry swaps the Rust servers for a guest server (kernel and services specs); gsv is reached only through sessions |
| Camera, chat image pools, event supervisor, 1111/1121 | Logged stubs with known reply shapes today; real behaviour command by command |
| 216 sparse map batches beyond decoding and validation | Ported with slice 7's model; extended if a title uses more fields |
| GPU-only memory in snapshots | A GPU snapshot (#12/#14) reads the host storage back |
| Splitting gsv's shared lock | The shared state is one struct behind one lock; splitting it per family later changes no handler |
