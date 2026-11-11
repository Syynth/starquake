# Kernel: object model and port plan

Status: **proposal, awaiting owner review** (issue #9). Builds on the execution model
(`execution-model.md`, also a proposal); where that changes, this follows.

`crates/kernel` re-implements the parts of the Horizon kernel a game can observe: threads and
scheduling, synchronisation, memory, handles, and the client side of IPC. Its behaviour is ported
from Eden (`eden/src/core/hle/kernel`, 183 files); its structure is not.

## What is built now, what is deferred, and what is not planned

Compatibility comes first (decision log, 2026-11-11): a feature may be deferred, but the design
must not rule it out. Starquake starts with one application process talking to services written in
Rust, so a large part of the kernel can wait. It is sorted into three groups:

| Built now | Deferred, and designed for | Not planned |
|---|---|---|
| Threads, the priority scheduler, core affinity, yields | **Server-side IPC**: `ReplyAndReceive`, `AcceptSession`, `CreateSession`, server ports, light sessions | Fibers, `cpu_manager`, `physical_core` (replaced by the execution model) |
| Events, `WaitSynchronization`, address arbiter, condition variables, lock arbitration with priority inheritance | **More guest processes**: real system modules, real applets, homebrew that hosts services; process creation calls | Horizon's internal allocators (slab heaps, page heaps): replaced by pool accounting |
| Address-space bookkeeping: regions, memory blocks, heap, map/unmap, permissions, shared and transfer memory | **Mapping IPC buffers into a server's address space** (HLE handlers read the client's memory directly) | Hardware-facing calls (I/O pools, interrupt events, device address spaces, secure monitor), unless a specific module justifies emulating the device behind them |
| Handle table, sessions with a Rust handler, named ports | Debugging system calls | |
| `GetInfo`, ticks, thread and process lifecycle | | |

Retail titles can't use the deferred calls: the Switch 2 titles Eden-NX2 examined are permitted
system calls 0x01–0x36 and 0x80 only, and the server side starts at 0x40. They matter for running
Nintendo's real system modules as a fallback and for homebrew that provides its own services.

### Rules that keep the deferred column cheap to add

1. **Sessions are modelled as Horizon models them:** a client end, a server end, and a queue of
   requests between them. A Rust handler is one kind of server end; a guest process calling
   `ReplyAndReceive` would be another. Nothing in the client path knows which it is.
2. **Nothing assumes a single process.** Processes live in the arena like any other object, each
   with its own address space, handle table and memory pool.
3. **Handle transfer is a kernel primitive** between any two handle tables, not a special case
   of calling a Rust handler.
4. **Memory blocks carry Horizon's full attribute set** (IPC lock count, device share count)
   from the first slice, even where nothing sets them yet, so mapping IPC buffers into a server is
   additive.
5. **Cores are configuration.** Application cores 0–5 are what exists at first; system cores can
   be added without touching the scheduler's logic.
6. **The system-call table lists every Horizon call** with a status: implemented, deferred or
   not planned. A deferred call returns "not implemented" and is logged, so a program that needs
   one says so in its trace.
7. **The service manager routes a name to a server**, which today is always a Rust factory.
   Letting a guest register a service later means adding one more kind of entry.

## Object model

### One arena behind one lock

Horizon's kernel objects are intrusively reference-counted and linked into each other's queues,
all guarded by the scheduler lock. Translating that literally into Rust means `Arc<Mutex<...>>`
everywhere, lock-order rules nobody can check, or `unsafe`.

Instead, everything the scheduler lock protects lives in **one struct owned by one mutex**, and
objects refer to each other by typed ids:

```rust
pub struct Kernel { state: Mutex<State>, cores: [CoreHandle; 6], /* immutable config */ }

struct State {
    threads: Slab<ThreadId, Thread>,        // run state, priority, affinity, wait reason, saved context
    objects: Slab<ObjectId, Object>,        // events, sessions, shared memory, ...
    run_queues: PriorityQueues,             // per core, 64 priorities, scheduled + suggested
    arbiter: BTreeMap<u64, WaitList>,       // address arbiter and condition-variable keys
    timers: BinaryHeap<Deadline>,
    processes: Slab<ProcessId, Process>,    // address space, handle table, pools
}
```

- A system call locks `State`, does its bookkeeping, and returns `Done`, `Wait` or `Yield`
  (execution model). Nothing else can observe a half-updated wait queue.
- No `unsafe` in the crate, no per-object locks, no lock ordering. Stale ids are caught by
  generation counters in the slabs.
- The whole kernel is a deterministic state machine: given the same calls in the same order it
  ends in the same state. Tests drive it directly without threads or a hypervisor.
- When the one lock becomes the bottleneck (Eden-NX2 notes, row 109), the address arbiter moves to
  its own sharded map. It is already a separate field with a narrow interface.

### A closed set of object kinds

```rust
enum Object { Event(Event), Session(Session), SharedMemory(..), TransferMemory(..), Thread(ThreadId), Process(ProcessId), ResourceLimit(..) }
```

An enum, not trait objects: the set is fixed by the system-call table, matches are exhaustive,
and there are no downcasts. Deferred kinds (server ports, light sessions) get their variants when
they are built; the compiler then points at every match that must handle them. Reference counts are explicit (handles and in-flight waits hold
them), so "closing the last handle destroys the object" is testable.

### Handles

Per process, Horizon's encoding (15-bit index, 15-bit generation) and the two pseudo-handles for
the current thread and process. Table size comes from the NPDM.

### Waiting and waking

A blocked thread records what it waits on and where its result goes:

```rust
enum Wait { Sync { objects: SmallVec<ObjectId>, .. }, Address { key: u64, .. }, CondVar { key: u64, lock: u64, .. }, Sleep, Reply { session: ObjectId } }
```

The waker removes it from the queue, writes the result into its saved `x0`–`x7`, and makes it
runnable; if that thread should now run on a core that is executing guest code, the waker
interrupts that core. Timeouts are entries in the timer heap carrying the thread's generation, so
a late timer for a thread that was already woken is ignored.

### Scheduler

Ported decision for decision from Eden's `k_priority_queue.h` and `k_scheduler.cpp`:

- Per-core queues of scheduled and suggested threads over 64 priorities.
- Choosing each core's next thread, and migrating a suggested thread to an idle core.
- The three yields, priority and core-mask changes, and thread activity (pause and resume).
- Priority inheritance for lock arbitration.

The scheduler talks to the outside through a small trait (`interrupt(core)`, `wake(core)`), so it
can be tested with a fake.

### Address space

Horizon's page-table semantics from `k_page_table_base.cpp` and `k_memory_block*.h`, without the
hardware part:

- A sorted map of memory blocks (state, permission, attributes, reference counts) with merge and
  split.
- Region layout for the 39-bit address space, sized from physical memory as Switch 2 expects.
- The operations system calls perform. Each one validates against the block map and then calls
  the execution model's `Translation` trait to make the change real.
- Pool accounting (application, applet, system, non-secure system) for the 12 GB layout. What
  `GetInfo` reports must match what was actually reserved, including memory the graphics server
  takes (rows 37, 75–80).

Region placement is **deterministic by default** (no randomisation) so traces and crash addresses
reproduce; a seed can turn randomisation on.

### IPC

A session has two ends and a request queue:

```rust
struct Session { queue: VecDeque<Request>, current: Option<Request>, server: ServerEnd, client_open: bool }
enum ServerEnd { Hle(Box<dyn SessionHandler>), /* later: Guest { .. } */ }
```

- `ConnectToNamedPort("sm:")` returns a handle to the client end of a session whose server end is
  the service manager.
- `SendSyncRequest` copies the 0x100-byte message out of the thread's TLS block and queues a
  request. If the session is idle the request starts at once; if another request is in flight
  (for example, waiting on a deferred reply), **the sender waits as a guest thread**, exactly as
  it would on a console. A core thread never waits on a host lock for a session.
- Starting a request on an HLE server end: parse the message with `starquake-ipc`, resolve copy
  and move handles against the caller's handle table, and call the handler **without holding
  the kernel lock**. Reply handles are inserted into the caller's table on the way back, the
  caller is woken, and the next queued request starts.
- Closing either end follows Horizon: queued and in-flight requests complete with the
  session-closed result.
- The handler interface is defined here (`SessionHandler`: parsed request in, `Done` or
  `Deferred` out) and implemented by the service framework (`services.md`).

A guest server end would take requests from the same queue with `ReplyAndReceive`; that, server
ports and buffer mapping are the deferred work.

### System calls

One `match` from system-call number to a typed function. The register convention (which
argument is in which register, which outputs go where) is taken from Eden's generated wrappers
in `svc.cpp`, as a table with a test per entry. An unknown or unimplemented call is logged in
the trace and returns Horizon's "not implemented" result; it never panics.

NPDM capabilities (allowed system calls, core mask, priority range) are checked and **logged**
when violated at first, then enforced once milestone 2 titles boot.

### Switch 2 specifics

`GetInfo` ids 29–33, `svc 0x80`, six application cores, the larger heap and alias regions, and
the 12 GB layout (scoping doc, "Process, memory, kernel"). Each lands with the port slice it
belongs to, citing its row.

### Recording and replay

The observability spec (`observability.md`, also a proposal) adds three requirements here: the
system-call dispatcher, scheduler and address space each report to one event recorder; the clock,
random numbers and host I/O are read through an injectable environment, never directly; and the
arena is serialisable, for snapshots. All three are cheap with this design and are part of slice 2
onwards.

## Port plan

Each slice is one issue with its own tests. Slices 1–5 touch only kernel bookkeeping and can run
in parallel once this spec and the execution model are approved; 6–10 build on them.

| # | Slice | Eden source | Test |
|---|---|---|---|
| 1 | Result codes, system-call ids and register convention table | `svc_results.h`, `svc.h`, `svc.cpp` | Table test per call |
| 2 | Slabs, typed ids, handle table | `k_handle_table.*`, `k_auto_object.h` | Encoding fixtures; generation reuse; property test against a model |
| 3 | Thread state, priority queues, scheduler | `k_priority_queue.h`, `k_scheduler.*`, `k_thread.*` | Invariants under random operations (highest-priority runnable thread runs, no thread on two cores, affinity respected); scenario tests for yields and migration |
| 4 | Events and `WaitSynchronization` | `k_event.*`, `k_readable_event.*`, `k_synchronization_object.*` | Wake order, cancellation, timeouts, signalled-before-wait |
| 5 | Address arbiter, condition variables, lock arbitration | `k_address_arbiter.*`, `k_condition_variable.*`, `k_thread.*` (waiters) | Every arbitration and signal type; priority inheritance chains |
| 6 | Memory blocks, regions, heap, map/unmap/permissions, `QueryMemory` | `k_memory_block*.h`, `k_page_table_base.*` | Model test against a naive page-by-page map; Horizon's error for each invalid request |
| 7 | Pools, accounting, `GetInfo` | `k_memory_manager.*`, `svc_info.cpp`, PR #4588's additions | Accounting adds up after random operations; Switch 2 ids |
| 8 | Process creation, TLS pages, thread lifecycle calls | `k_process.*`, `k_thread_local_page.*`, `svc_thread.cpp` | Create, start, exit, join; TLS slot reuse |
| 9 | Sessions (both ends and the request queue), named ports, `SendSyncRequest` with handle translation | `k_session.*`, `k_client_session.*`, `k_server_session.*` (queueing and close semantics only), `svc_ipc.cpp`, `hle_ipc.cpp` | Fake handler; a second sender queues behind a deferred request; deferred reply wakes the caller; closing either end; handle copy and move |
| 10 | Core loop and timers | none (new) | Scripted fake vCPUs drive six cores deterministically |

Later slices: shared and transfer memory, `MapPhysicalMemory`, resource limits, thread activity
and context, code memory. Deferred until something needs them: the server-side calls, guest
process creation, and IPC buffer mapping.

## Decisions asked of the owner

1. **One arena behind one lock, objects referenced by id.** Recommended over mirroring
   Horizon's linked, reference-counted objects: no `unsafe`, deterministic tests, and the known
   contention point can be split out later.
2. **A closed `enum` of object kinds** rather than trait objects.
3. **Server-side IPC, guest-hosted services and real system modules are deferred, not excluded.**
   The seven rules above keep them additive. (Revised after owner review: the first draft ruled
   them out.)
4. **Deterministic address-space layout by default.**
5. **Capabilities are logged first, enforced later.**
