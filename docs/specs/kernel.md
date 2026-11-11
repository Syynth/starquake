# Kernel: object model and port plan

Status: **approved by the owner, 2026-11-11** (issue #9). Builds on the execution model
(`execution-model.md`).

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
   requests between them, with a server that owns many server ends. A Rust handler is one kind of
   server; a guest process calling `ReplyAndReceive` would be another. Nothing in the client path
   knows which it is.
2. **Nothing assumes a single process.** Processes live in the tables like any other object, each
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

### Tables behind a dispatcher, one lock to start

Horizon's kernel objects are intrusively reference-counted and linked into each other's queues.
Translating that literally into Rust means `Arc<Mutex<...>>` on every object, lock-order rules
that depend on the data, or `unsafe`.

Instead, kernel state lives in a few **tables**, and objects refer to each other by typed ids:

| Table | Holds | Horizon's equivalent lock |
|---|---|---|
| `Scheduler` | Threads (run state, priority, affinity, wait reason, saved context), per-core run queues, wait lists, events and sessions, the address-arbiter map, timers | The scheduler lock (`k_scheduler_lock.h`) |
| `AddressSpace`, one per process | Memory blocks, regions, heap; plus the memory pools | Each page table's own lock (`m_general_lock` in `k_page_table_base.h`) |
| `HandleTable`, one per process | Handle slots | The handle table's own lock (`k_handle_table.h`) |
| `Processes` | Per-process facts fixed at creation (capabilities, ids) | None needed |

The split follows Horizon's own: it has one lock for everything about thread state and separate
locks for each memory map and handle table.

**A system call declares the tables it needs in its signature**, and receives nothing else:

```rust
fn set_heap_size(mem: &mut AddressSpace, size: u64) -> SvcResult<u64>;
fn signal_event(sched: &mut Scheduler, handles: &HandleTable, handle: Handle) -> SvcResult<()>;
fn get_system_tick(env: &dyn Env) -> SvcResult<u64>;            // no table at all
```

The **dispatcher** is the only code that locks. It acquires what the signature asks for, in one
fixed order (handle table, then address space, then scheduler), calls the function, and assigns
the event's sequence number before releasing. System calls never lock, so they can't deadlock,
and the compiler rejects a call that reaches for a table it didn't declare.

**To start, one mutex guards all the tables.** The dispatcher takes it and hands each call
disjoint borrows of the tables it declared. That keeps the first implementation simple and gives
one global order of events for free. Splitting later, when a measurement says so, changes the
dispatcher and nothing else:

1. Give `AddressSpace` its own lock, so a long memory operation doesn't stall scheduling.
2. Make `HandleTable` a reader-writer lock; almost every call only reads it.
3. Shard the address arbiter's wait lists by address, so the hot `WaitForAddress` and
   `SignalToAddress` pair (about 1,400 calls a frame) holds the scheduler lock only for the
   final wake-up. Table-level splitting alone doesn't help these calls: every wake-up touches
   the run queues.

After a split, the sequence number taken while holding a call's locks still gives one order that
is consistent with every table's own order, so replay keeps working.

What this buys, at any stage:

- No `unsafe` in the crate and no per-object locks. Stale ids are caught by generation counters.
- The kernel is a deterministic state machine: the same calls in the same order end in the same
  state. Tests drive it directly, without threads or a hypervisor.
- Tables are plain data, so snapshots serialise them directly (taking every lock at once).
- Separate table structs can be edited at the same time, which removes most of the
  borrow-checker friction a single struct would cause.

The dispatcher reports how long calls wait for each lock in the run's summary, so the decision
to split is made from measurements. Eden-NX2 saw three of six cores waiting on its scheduler
lock (row 109); whether starquake does is not yet known.

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

Modelled as Horizon models it: a **server** owns the server ends of many sessions and takes
requests from them one at a time.

```rust
struct Session { server: ServerId, in_flight: Option<Request>, waiting: VecDeque<Request>, client_open: bool }
struct Server  { kind: ServerKind, ready: VecDeque<SessionId>, running: u8, workers: u8 }
enum ServerKind { Hle(Box<dyn SessionHandler>), /* later: Guest { .. } */ }
```

- `ConnectToNamedPort("sm:")` returns a handle to the client end of a session on the service
  manager's server.
- `SendSyncRequest` copies the 0x100-byte message out of the thread's TLS block and queues a
  request on the session. **The sender then waits as a guest thread** whenever the request can't
  start at once, exactly as it would on a console. A core thread never waits on a host lock.
- A request starts when its session has nothing in flight **and** its server has a free worker.
  Servers have one worker by default, so one handler per server runs at a time (`services.md`,
  "Concurrency"); a deferred request is in flight on its session but doesn't hold the worker.
- Starting a request on an HLE server: parse the message with `starquake-ipc`, resolve copy and
  move handles against the caller's handle table, and call the handler **without holding the
  kernel lock**. Reply handles are inserted into the caller's table on the way back, the caller
  is woken, and the core that finished starts the server's next ready request.
- Closing either end follows Horizon: waiting and in-flight requests complete with the
  session-closed result.
- The handler interface is defined here (`SessionHandler`: parsed request in, `Done` or
  `Deferred` out) and implemented by the service framework.

A guest server would take requests from the same queues with `ReplyAndReceive`, which waits on
many sessions at once; that, server ports and buffer mapping are the deferred work.

### System calls

One `match` from system-call number to a typed function. The register convention (which
argument is in which register, which outputs go where) is taken from Eden's generated wrappers
in `svc.cpp`, as a table with a test per entry. An unknown or unimplemented call is logged in
the trace and returns Horizon's "not implemented" result; it never panics.

NPDM capabilities (allowed system calls, core mask, priority range) are checked and **logged**
when violated at first, then enforced once milestone 2 titles boot.

### Guest time

The kernel owns guest ticks. The environment returns host nanoseconds (a distinct type), and the
kernel converts them with the process's counter frequency: 31.25 MHz for a Switch 2 process,
19.2 MHz for a Switch 1 process. `core` hands the same host and guest frequencies to the loader's
counter patcher, and a test checks that a tick read through a patched instruction and one read
through `GetSystemTick` agree. A replay whose recorded clock runs out or disagrees stops the run
with a replay error; a guest never sees that as a result code. (Added by the lead under
delegation, 2026-11-11, after wave 3 found that no crate owned the conversion.)

The system-call table has four statuses: implemented, planned (its slice hasn't landed),
deferred and not planned.

### Switch 2 specifics

`GetInfo` ids 29–33, `svc 0x80`, six application cores, the larger heap and alias regions, and
the 12 GB layout (scoping doc, "Process, memory, kernel"). Each lands with the port slice it
belongs to, citing its row.

### Recording and replay

The observability spec (`observability.md`) adds three requirements here: the
system-call dispatcher, scheduler and address space each report to one event recorder; the clock,
random numbers and host I/O are read through an injectable environment, never directly; and the
tables are serialisable, for snapshots. All three are cheap with this design and are part of slice 2
onwards.

## Port plan

Each slice is one issue with its own tests. Slices 1–5 touch only kernel bookkeeping and can run
in parallel once this spec and the execution model are approved; 6–10 build on them.

**Slices 1–3 are a trial of this design.** The scheduler is the most pointer-heavy part of
Eden's kernel. If it ports cleanly onto tables and ids, the patterns it establishes (editing two
rows at once, walking wait lists by id) become the house style for the rest. If it doesn't, the
design is revisited before slices 4–10 depend on it.

| # | Slice | Eden source | Test |
|---|---|---|---|
| 1 | Result codes, system-call ids, register convention table, and the dispatcher with table parameters | `svc_results.h`, `svc.h`, `svc.cpp` | Table test per call; a call can't reach a table it didn't declare (compile-fail test) |
| 2 | Slabs, typed ids, handle table | `k_handle_table.*`, `k_auto_object.h` | Encoding fixtures; generation reuse; property test against a model |
| 3 | Thread state, priority queues, scheduler | `k_priority_queue.h`, `k_scheduler.*`, `k_thread.*` | Invariants under random operations (highest-priority runnable thread runs, no thread on two cores, affinity respected); scenario tests for yields and migration |
| 4 | Events and `WaitSynchronization` | `k_event.*`, `k_readable_event.*`, `k_synchronization_object.*` | Wake order, cancellation, timeouts, signalled-before-wait |
| 5 | Address arbiter, condition variables, lock arbitration | `k_address_arbiter.*`, `k_condition_variable.*`, `k_thread.*` (waiters) | Every arbitration and signal type; priority inheritance chains |
| 6 | Memory blocks, regions, heap, map/unmap/permissions, `QueryMemory` | `k_memory_block*.h`, `k_page_table_base.*` | Model test against a naive page-by-page map; Horizon's error for each invalid request |
| 7 | Pools, accounting, `GetInfo` | `k_memory_manager.*`, `svc_info.cpp`, PR #4588's additions | Accounting adds up after random operations; Switch 2 ids |
| 8 | Process creation, TLS pages, thread lifecycle calls | `k_process.*`, `k_thread_local_page.*`, `svc_thread.cpp` | Create, start, exit, join; TLS slot reuse |
| 9 | Sessions and servers (both ends, request queues, workers), named ports, `SendSyncRequest` with handle translation | `k_session.*`, `k_client_session.*`, `k_server_session.*` (queueing and close semantics only), `svc_ipc.cpp`, `hle_ipc.cpp` | Fake handler; a second sender on a session queues behind a deferred request; two sessions of one server don't run handlers at once; deferred reply wakes the caller; closing either end; handle copy and move |
| 10 | Core loop and timers | none (new) | Scripted fake vCPUs drive six cores deterministically |

Later slices: shared and transfer memory, `MapPhysicalMemory`, resource limits, thread activity
and context, code memory. Deferred until something needs them: the server-side calls, guest
process creation, and IPC buffer mapping.

## Decisions

Approved by the owner on 2026-11-11 as written. Where a decision names a recommendation, the
recommendation stands.

1. **Tables behind a dispatcher, objects referenced by id, one lock to start.** System calls
   declare the tables they touch; only the dispatcher locks. Recommended over mirroring Horizon's
   linked, reference-counted objects: no `unsafe`, deterministic tests, and the single lock can
   be split table by table later without touching the system calls. (Revised after owner review:
   the first draft had one undivided arena.)
2. **A closed `enum` of object kinds** rather than trait objects.
3. **Server-side IPC, guest-hosted services and real system modules are deferred, not excluded.**
   The seven rules above keep them additive. (Revised after owner review: the first draft ruled
   them out.)
4. **Deterministic address-space layout by default.**
5. **Capabilities are logged first, enforced later.**
