# Execution model: vCPUs, guest memory and kernel threading

Status: **proposal, awaiting owner review** (issue #7). Nothing here is implemented beyond the
`Vcpu`/`Backend`/`GuestMemory` traits and the hypervisor backend's bring-up memory.

This spec decides how guest code, guest memory and the HLE kernel fit together. It is written for
the native backends first (Hypervisor.framework, KVM) and says what a JIT backend must provide to
fit the same interfaces.

## Summary

- **Six core threads, one vCPU each.** A guest thread is saved register state that the scheduler
  loads into a core's vCPU. A context switch is a register swap, not a host thread switch.
- **HLE code never blocks a core thread.** System calls return either a result or "this guest
  thread now waits on X". Services that can't answer immediately defer their reply.
- **We own the guest's stage-1 page tables.** They use a 4 KiB granule and map guest virtual
  addresses onto a few large, fixed stage-2 regions. All fine-grained mapping, protection and
  write tracking happens in stage 1, which is plain data we can build and test off-hardware.
- **Guest memory is never a Rust slice.** The guest can write to it at any time, so host code
  copies in and out or uses a raw view type.

## Guest threads on vCPUs

### Core threads

One host thread per application core (0–5). Each creates and owns one vCPU for its whole life;
Hypervisor.framework requires a vCPU to be run by the thread that created it. Titles pin their
threads to specific cores and size their job systems by core count (Eden-NX2 notes, rows 77–79),
so the six cores are real, not folded.

```text
core thread:
  loop
    thread = scheduler.pick(core)        // parks the host thread when nothing is runnable
    vcpu.load(thread.context)
    loop
      match vcpu.run()
        Svc(n)        -> match kernel.svc(thread, n, vcpu)
                           Done      -> continue          // results already in x0..x7
                           Wait      -> break             // thread is now blocked
                           Yield     -> break
        DataAbort     -> kernel.fault(thread, ...) -> continue | break (thread killed)
        Undefined     -> kernel.undefined(thread, ...)
        Interrupted   -> break if scheduler wants this core, else continue
    vcpu.save(thread.context)
```

- **System calls touch only the registers they need.** The vCPU keeps the guest's state across a
  system call; the handler reads arguments from `x0`–`x7` and writes results back. The full
  context (31 general registers, SP, PC, PSTATE, thread pointers, 32 vector registers, FPCR/FPSR)
  is saved and loaded only on a context switch. The `Vcpu` trait grows single-register accessors
  for this.
- **Preemption.** When a wake-up makes a higher-priority thread runnable on a core that is
  running guest code, the waker interrupts that core's vCPU (`hv_vcpus_exit`); it returns
  `Exit::Interrupted` and reschedules. The same mechanism implements time slicing and shutdown.
- **Idle.** A core with nothing runnable parks on a condition variable and is woken by the
  scheduler.
- **Timers.** One host timer thread holds a deadline heap (sleep and wait timeouts, vsync
  events). When a deadline fires it wakes the guest thread through the scheduler like any other
  waker.

### Scheduling

Port Horizon's scheduling *decisions* from Eden (`k_scheduler`, `k_priority_queue`,
`k_thread`): per-core priority queues, affinity masks, ideal core, migration of runnable threads
to idle cores, and the yield variants. Don't port Eden's fibers or its `cpu_manager`: under a
hypervisor they have no job.

Start with one scheduler lock, as Horizon and Eden do. Eden-NX2 saw three of six cores blocked on
it under a real title (row 109), so kernel objects are written behind an interface that lets the
hot pair (`WaitForAddress`/`SignalToAddress`, about 1,400 calls a frame) move to per-bucket locks
later without changing callers.

### System calls never block the host thread

A system call handler is an ordinary function that returns one of:

- `Done`: results are in the guest's registers; resume.
- `Wait(reason)`: the guest thread was queued on a kernel object (synchronisation object,
  address arbiter key, condition variable, sleep deadline, IPC reply). The waker later writes the
  result into the thread's saved `x0`–`x7` and makes it runnable. This is how the real kernel
  does it, and it is what makes a context switch a register swap.
- `Yield`: reschedule.

### IPC and services

`SendSyncRequest` to an HLE service is handled **inline on the calling core thread**: parse the
message with `starquake-ipc`, call the service's handler, write the reply, return `Done`. No
thread hop on the common path, and traces are in call order.

A handler that can't answer yet (vi2 present waiting for a frame fence, a read waiting on host
I/O) returns `Deferred` and keeps a reply token. The guest thread waits exactly as it would on a
real server. Whoever finishes the work (GPU thread, timer thread, an I/O worker) completes the
token, which writes the reply into the thread's IPC buffer and wakes it. Handlers must never
sleep, wait on a host lock for long, or do blocking I/O.

Services that are active on their own (audio renderer, vsync, input polling) are host threads.
They interact with guest threads only by signalling kernel objects through thread-safe handles.

## Guest memory

### Three layers

1. **Backing memory.** Host allocations that stand for the console's RAM, split into Horizon's
   pools (application, applet, system, non-secure system) using the 12 GB layout. Pool accounting
   must match what titles are told by `svcGetInfo`: every memory-shortage crash in Eden-NX2's
   notes (rows 75–80) was accounting, not emulation.
2. **Address space.** Per process: Horizon's page-table semantics ported from Eden's
   `k_page_table_base` and `k_memory_block`: regions (code, alias, heap, stack, TLS), memory
   states, permissions and attributes, and the operations system calls perform on them. Pure
   bookkeeping, no hardware.
3. **Translation.** What makes the bookkeeping real for guest code. One trait, two families of
   implementation:

```rust
trait Translation {
    fn map(&self, va: u64, backing: BackingRange, perms: Perms) -> Result<(), MapError>;
    fn unmap(&self, va: u64, size: u64) -> Result<(), MapError>;
    fn protect(&self, va: u64, size: u64, perms: Perms) -> Result<(), MapError>;
}
```

### Native backends: stage 1 is ours, stage 2 is coarse

- **Stage 2 (guest-physical → host)** is set up once: each backing pool is mapped at a fixed
  guest-physical base as one large region. Apple hosts use 16 KiB pages, so stage 2 can't express
  4 KiB mappings; it doesn't need to.
- **Stage 1 (guest-virtual → guest-physical)** uses a 4 KiB granule and a 39-bit address space,
  matching what titles expect (`svcGetInfo` 33 reports 0x1000). The tables live in their own
  backing region and are written directly by the host. `tools/hvprobe` verified that 4 KiB pages
  inside one 16 KiB host page map independently.
- `Stage1Tables` is a plain data structure: given `map`/`unmap`/`protect` calls it writes table
  entries into a byte region. A software walker reads them back. Both are safe Rust, testable on
  any host, and shared by the Hypervisor.framework and KVM backends. A hardware test on a Mac
  cross-checks the walker against a real vCPU.
- **TLB invalidation** has to run inside the guest: the host can't flush a guest's stage-1 TLB.
  The EL1 shim gets an entry point that executes the pending invalidations before returning to
  EL0. For changes that affect other cores, either the broadcast forms (`TLBI ...IS`) reach every
  vCPU of the VM, or each core is interrupted and flushes before resuming. **Which one holds
  under Hypervisor.framework is unverified** and is the first follow-up measurement.
- **Faults** arrive as `Exit::DataAbort`. The kernel decides: fill in a lazily backed page and
  resume, deliver a write-tracking notification and resume, or kill the thread and report.
- **Pointer authentication keys** are per process, generated by the kernel, and set on every vCPU
  when the process starts running on it.

### Host access to guest memory

The kernel reads IPC buffers, the GPU reads command buffers and textures at guest virtual
addresses (on Switch 2, GPU addresses equal CPU addresses), and services copy data in and out.
All of it goes through `GuestMemory`, which resolves addresses with a host-side shadow of the
address space, not by walking the hardware tables.

Guest threads can write to any of this memory at any moment, so host code must not hold `&[u8]`
or `&mut [u8]` into it: that would be a data race. The interface offers copies (`read`/`write`)
and, for hot paths, a raw view type whose accessors use volatile or atomic reads. Only the
memory crate contains the `unsafe` for this.

### Write tracking for the GPU caches

The texture and buffer caches need to know when the guest CPU writes to memory they have
uploaded. Protect those pages read-only in stage 1 (4 KiB granularity); the first write faults,
the kernel notifies the cache, restores write permission and resumes. Cost is one fault per page
per tracking period, about a system call's round trip (~0.8 µs measured).

### Sparse and lazy memory

Titles reserve far more address space than they use (two 5.7 GiB GPU regions in one case). A
reservation creates address-space bookkeeping but no stage-1 entries; backing pages are
allocated on first touch (fault) or when the graphics server maps them.

## What other backends must provide

- **KVM on arm64:** the same design. Stage 2 becomes memory slots; `Stage1Tables`, the EL1 shim
  and the fault handling are shared.
- **JIT (x86-64 hosts):** implements `Translation` with a software MMU and inline fast paths,
  reports the same `Exit` values, counts the guest counter itself, and emulates pointer
  authentication (at minimum as the identity), LSE atomics and RCpc loads. It can also offer a
  deterministic single-threaded mode for tests, which the native backends can't.

## The `Vcpu` trait after this spec

```rust
trait Vcpu {
    fn run(&mut self) -> Result<Exit, Error>;
    fn reg(&self, index: u8) -> Result<u64, Error>;            // x0..x30
    fn set_reg(&mut self, index: u8, value: u64) -> Result<(), Error>;
    fn save(&self, context: &mut GuestContext) -> Result<(), Error>;   // full state incl. vectors
    fn load(&mut self, context: &GuestContext) -> Result<(), Error>;
    fn set_address_space(&mut self, root: TableRoot, asid: u16) -> Result<(), Error>;
    fn set_pauth_keys(&mut self, keys: &PauthKeys) -> Result<(), Error>;
    fn interrupter(&self) -> Interrupter;                       // Send + Sync
}

enum Exit {
    Svc(u16),
    DataAbort { address: u64, write: bool },
    InstructionAbort { address: u64 },
    Undefined { pc: u64 },
    Interrupted,
}
```

## To verify before building on this

Measurements to add to `tools/hvprobe`, in order of how much they could change the design:

1. Do broadcast TLB invalidations executed on one vCPU take effect on the others?
2. Cost of a full context switch (save and load of all general and vector registers).
3. Cost of a stage-1 write-protection fault round trip, and of the EL1 flush entry point.
4. System call round-trip cost with six vCPUs running at once.

## Decisions asked of the owner

1. **Deferred replies instead of service threads** for services that wait. Recommended: it keeps
   one mechanism (guest threads wait on kernel objects) and makes traces deterministic in order.
2. **One scheduler lock to start**, with the address arbiter designed to split out later.
   Recommended: correctness first; the hot spot is known and isolated.
3. **No deterministic replay on native backends.** Tests compare ordered traces per guest thread
   and avoid depending on cross-thread timing. A deterministic mode waits for the JIT backend.
4. **Stage-1 tables as the single place for fine-grained mapping**, with stage 2 fixed. The
   alternative, tracking through stage 2, can't do 4 KiB on Apple hosts.
