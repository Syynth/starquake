# hvprobe

A small Rust program that checks whether Apple's Hypervisor.framework (HVF) can run Switch 2
guest code natively. It creates a VM, builds 4 KiB stage-1 page tables, installs an EL1 vector
table, and runs hand-written guest tests at EL1 and EL0. The guest code is in `src/guest.rs`;
the Hypervisor.framework bindings are in `src/hv.rs`.

```sh
cargo run --release -p hvprobe
```

The workspace's cargo runner (`tools/macos-runner.sh`) signs the binary with the hypervisor
entitlement first.

## Results: Apple M2, macOS 27.0.1 (2026-11-11)

| Question | Result | Consequence |
|---|---|---|
| Guest IPA (physical address) space | default and max **36 bits (64 GiB)** | Enough for the 12 GB layout. The emulator's large GPU/SMMU spaces are its own bookkeeping, not IPA. |
| `CNTFRQ_EL0` / counter rate | **24 MHz**, measured 23.999–24.000 MHz | Switch 2 expects 31.25 MHz. HVF can offset the virtual counter but not scale it. |
| `CNTPCT_EL0` (the physical counter Nintendo's SDK reads) | **Always traps to the host** (EC 0x18), at EL1 and EL0, even with `CNTKCTL_EL1.EL0PCTEN=1`; **~760 ns per read** | Unpatched, Mario Kart World's ~17M reads/s would need ~13 host cores. |
| Same read with `EL0PCTEN=0`, so it traps to the guest's own EL1 handler | **~26.5 ns per read** | ~45% of a core at 17M/s. Works as a fallback for unpatched sites, but not as the main path. |
| Load-time patch: `mrs xN, cntpct_el0` → `bl` to a stub that reads `cntvct_el0` and rescales to 31.25 MHz | **~0.88 ns per read** | ~1.5% of a core. **Patching is the plan.** |
| EL0 `svc` → guest EL1 vector → `hvc` → host → `eret` | **~0.76 µs per round trip** | ~1.1 ms per frame at Mario Kart World's ~1,400 SVCs/frame. Acceptable to start; worth a fast path for `WaitForAddress`/`SignalToAddress` later. |
| PAuth visible to the guest | `ID_AA64ISAR1_EL1`: API=4 (PAuth2 + FPAC), GPI=1 (PACGA); implementation-defined algorithm | `RETAA/RETAB`, `PAC*`, `AUT*`, `PACGA` run natively. No patching needed. |
| PAC keys settable from the host | Yes (`hv_vcpu_set_sys_reg` APIA/APIB key registers); sign/auth round trip correct | The HLE kernel sets per-process keys on every vCPU at context switch. |
| PAC values identical across vCPUs, E-cores and P-cores | Yes, 200 runs on fresh vCPUs | Guest threads can migrate between vCPUs freely. |
| Failed `AUT*` | FPAC: synchronous exception EC 0x1C to guest EL1 | Report it as a guest crash. On console this would also fault. |
| `LDAPR`, `CAS`, `LDADD`, `SWP` at EL0 | Correct, no traps | The ~200k ARMv8.1–8.4 instructions per title cost nothing. |
| MTE `STG` | UNDEFINED, EC 0x0 to guest EL1; skipping it works | Handle it in the guest EL1 handler as a no-op (one site per title), or patch it at load time. |
| 4 KiB stage-1 pages on a 16 KiB host | Works: two 4 KiB pages remapped out of order inside one 16 KiB host page read correctly | The guest's 4 KiB granule (`svcGetInfo` 33) is fine. Only stage-2 (IPA → host) mappings need 16 KiB alignment. |

### TLB invalidation across vCPUs (added 2026-11-11)

Two vCPUs share one set of stage-1 tables. The host repoints a page-table entry without flushing
while vCPU B keeps reading that page inside the guest. 40 rounds, run twice:

| Case | Time until B sees the new page |
|---|---|
| No flush at all | median 130–250 ms, up to ~1 s (B only notices when the host happens to interrupt it) |
| vCPU A executes `tlbi vmalle1is` at EL1 | 42 ns – 7 µs after A's flush returns |
| vCPU exits to the host and re-enters after the change | Sees the new page immediately in this test; don't rely on it |

Consequence: **broadcast TLB invalidation works across vCPUs.** After changing stage-1 tables,
one vCPU running the EL1 flush entry point is enough; other cores don't need to be interrupted.
A change must never be left unflushed: a running vCPU keeps a stale translation for a long time.

### Instruction cache and host writes (added 2026-11-11)

Hypervisor.framework does not keep the guest's instruction fetches coherent with stores the host
makes to guest RAM. A probe that rewrote `movz x0, #i; svc #i` in place between runs on one vCPU
ran the old instructions in 199,949 of 200,000 iterations (the review that found it); calling
`sys_icache_invalidate` (libSystem, `<libkern/OSCacheControl.h>`) on the written host range
brought that to 0 of 200,000. The backend's integration tests reproduce it: without the call the
second rewrite already runs the first program.

Consequence: **every host write to guest RAM must be followed by `sys_icache_invalidate` on the
written range.** The backend does this in its single write path (`HostRegion::write`), so
`GuestMemory::write` never leaves stale instructions. The same stale instructions also showed up
as rare failures across tests that create and destroy VMs in one process (3 of 240 full runs of the
test binary on a loaded machine, with signatures such as a data abort at address 0 or the wrong
`svc` immediate); they were gone after the invalidation (0 of 1,200 loaded runs). Jumping into RAM
that was never written, in a VM created right after one that executed code at the same address,
ran zeroes (a fault) every time in 1,500 trials, so no invalidation of untouched memory is done.

### SMCCC: `hvc #0` is not a plain exit (added 2026-11-11)

The EL1 shim reports an EL0 exception to the host with an `hvc`. Hypervisor.framework treats
**`hvc #0` as an SMCCC call**: when `w0` looks like a function id it knows, it answers in the
kernel, writes its result into `x0`-`x3`, and resumes the guest **without exiting to the host**.
The shim's `eret` then returns to EL0 just past the `svc`, which executes whatever follows.

| Observation | Value |
|---|---|
| `svc #0` with `x0` set to each possible top byte of `w0` (low 24 bits random, 400 each) | only top byte `0xc1` was swallowed (190 of 400); all others exited normally |
| What the guest sees after a swallowed `hvc #0` | `x0` = `0xffffffff`, or `x0`,`x1` = `1`,`0`, or four unrelated words in `x0`-`x3`; `ESR_EL1` = `0x2000000` (an UNDEFINED taken at the instruction after the `svc`) |
| The same `svc` with the shim using `hvc #2` | `x0` top byte `0xc1`: 0 of 300,000; every top byte exits normally |
| Odds for a guest that reads a counter into `x0` just before an exit | about 1 in 256: a 24 MHz count has top byte `0xc1` in its low word for 2^24 ticks (0.7 s) every 2^32 ticks (179 s), which is the 0.3 to 0.5% flake rate of the counter test and why failures came in bursts |

Only the top byte was scanned, so other function ids may be handled too. Consequence: **never
forward an exception with `hvc #0`**, and never rely on `x0` at the point of the `hvc`: `x0` is
the guest's own value (an `svc` argument) there. The shim uses `hvc #2` (`HVC_EL0_EXCEPTION`);
`hvc #1` marks a vector the shim doesn't use. Real guests hit this on any `svc` whose first
argument has top byte `0xc1` in its low word.

### Stage-1 tables under a running guest (added 2026-11-11)

Measured by the hypervisor backend's hardware tests (`crates/cpu/tests/hvf.rs`,
`crates/cpu/src/hvf/hardware.rs`), not by the probe, on the same Apple M2 with macOS 27.0.1.
The guest runs at EL0 on tables built by `starquake_memory::stage1::Stage1Tables` through
`TTBR0_EL1`; the EL1 shim is in the upper half through `TTBR1_EL1` (`T0SZ` = `T1SZ` = 25).

| Question | Result | Consequence |
|---|---|---|
| Address-space id width a guest gets | **8 bits**: `ID_AA64MMFR0_EL1` reads `0x10000f100001` in the guest (`ASIDBits` = 0). Two address spaces whose ids differed only above bit 7 (`0x0100`, `0x0200`) shared cached translations on the first switch | At most 256 ids. The spec's interface takes 16 bits; the backend reports the width and refuses a wider id. The kernel must hand out ids within it and flush before reusing one. |
| Do cached translations survive an exit to the host and a change of `TTBR0_EL1`? | **Yes.** A vCPU switched to an empty table under the same id kept running the old address space's code | An id must not be given to other tables without a flush. Switching between ids with no flush works: 200 switches between two address spaces mapping one address to different pages, each read its own. |
| Non-global pages (`nG`), execute-only pages (`AP` = 0b00 without `UXN`), `PXN` on every guest page | All behave as the Arm manual says; the software walker and the vCPU agree for all six permissions and three kinds of access, and for addresses missing at walk levels 1, 2 and 3 | The parts of `Stage1Tables` that hvprobe never wrote are now run on hardware. |
| Syndrome of a stage-1 fault from EL0 | Exactly class, `IL`, `WnR` and the fault status code; every other `ISS` bit zero | A backend can build the same value for a software MMU. |
| An address between the two halves (`0x80_0000_0000` to `0xffff_ff7f_ffff_ffff`) | Translation fault reported at **level 0**, for load, store and fetch | What the software walker calls out of range. |
| `tlbi vae1is` / `vale1is` / `aside1is` executed at EL1 by one vCPU | A second vCPU spinning in the guest on the unmapped page faulted within the test's bound (2,000,000 loop iterations after the flush returned) in 40 of 40 rounds. With the flush taken out it ran on for 270 million iterations | By-address broadcast invalidation works across vCPUs, as `vmalle1is` did above. |
| The same with the local form, `tlbi vae1` | The second vCPU was stopped just the same (40 of 40) | Hypervisor.framework appears to make a guest's local invalidations broadcast, so this test cannot tell the two forms apart. The backend uses the broadcast forms regardless. |
| Does a vCPU that stays in the guest keep its cached translation until the flush? | **Not on a busy host.** In the broadcast test above, with every core busy with other work, the spinning vCPU often faulted before the flush had run, at level 2 or 3 where the finished unmap says level 1 (32 of 40 runs had at least one such round: 17 at level 2, 15 at level 3). Why it walks again is not measured (a guess: the host descheduled its thread) | An unmap is visible to other vCPUs from its first descriptor write, in whatever state the walk finds, not only once the flush has run. Nothing may assume an old translation stays usable until the flush. |
| Leaf-only invalidation (`vale1is`) after a page's level-3 and level-2 tables were unlinked | The next access faulted at **level 3** where the tables say level 1, on the vCPU that flushed and on another: both still reached the unlinked level-3 table | Cached walk entries are real. Unlinked table pages need the every-level form (`vae1is`), and must not be reused before it has run. |
| Stage-2 write protection of the translation tables and the shim | A guest store to a page stage 2 maps read-only exits to the host (`ESR_EL2` class 0x24) with the exact guest-physical address; a guest load works, and so do the MMU's own table walks | Tables and shim are mapped without write permission; a mapping that wrongly pointed at them could not change them. |
| A 64-bit descriptor written by the host as two halves (2,000 spin-loop iterations apart) while a vCPU walks to it | The vCPU stopped with a stage-2 fault at once: the half-written descriptor named `0x1000a000`, the low half of a page at `0x1_1000_a000` | Descriptors are written with one atomic 64-bit store. Guest RAM sits above 4 GiB so that a torn write cannot go unnoticed. |
| The shim's page made accessible to EL0 (`AP` = 0b01) | The vCPU never comes back: a page EL0 can write is not executable at EL1, so the shim faults into itself forever | The shim's mapping must stay EL1-only; the golden test of the static tables guards it. |

### Wait hints, FP/SIMD state and PSTATE (added 2026-11-12)

Measured by the same hardware tests on the same Apple M2 (`crates/cpu/tests/hvf.rs`).

| Question | Result | Consequence |
|---|---|---|
| EL0 `wfi` with `SCTLR_EL1.nTWI` set | Exits to the host: `ESR_EL2` `0x07e0_0000` (class 0x01, `TI` = `wfi`), the vCPU at EL0 on the `wfi` | The backend steps over it and enters the guest again. |
| EL0 `wfe` with `SCTLR_EL1.nTWE` set, the event already used up by `sevl; wfe` | Came back in the guest, no exit | Handled like `wfi` all the same, should another host trap it. |
| V0-V31 through `hv_vcpu_get/set_simd_fp_reg` | Every byte of all 32 round-trips both ways, and a context moved to a vCPU on another thread keeps them | The vector is passed by value in `q0`; the backend makes that call with inline assembly because Rust cannot yet pass vectors to foreign functions. |
| `FPCR`, `FPSR` with every bit written | The guest reads `0x07ff_9f00` and `0x0800_009f`; after a run the host reads the same. Before a run the host reads back the low 32 bits it wrote | The kept `FPCR` bits are Eden's `FPCR::mask` (`0x07FF9F00`, `eden/src/dynarmic/src/dynarmic/common/fp/fpcr.h`). The trap enables (`FPCR` bits 15, 12:8) are among them: a guest that sets `DZE` and divides by zero takes class 0x2c and crashes, where Eden never delivers a trapped FP exception (`FPProcessException` in `eden/src/dynarmic/src/dynarmic/common/fp/process_exception.cpp` asserts the enable is clear). A listed divergence; what the Cortex-A78C does is a guess. |
| `PSTATE.DIT`, `PSTATE.SSBS` | `msr dit`, `msr ssbs` work at EL0; both bits show in `SPSR_EL1` at an exit and reach the guest from it | A context keeps NZCV, DIT and SSBS. |

What this doesn't cover yet: VM-exit cost with six vCPUs running at once, full context-switch cost,
`hv_vcpus_exit` for preempting a vCPU, stage-2 remapping cost (lazy backing of sparse regions),
and exception-level details of a real HLE kernel (TPIDRRO_EL0, EL0 data aborts for
fastmem-style tracking).

## What it means across backends

Counter reads need load-time patching under HVF. The same patch works under KVM on ARM64 Linux,
where physical counter access may also trap depending on the host, and it's harmless under an
x86-64 JIT, which can read its own clock directly. So the load-time **code patcher** belongs in
the shared loader, not in one backend. Each backend supplies its own stub:

- **HVF / KVM:** a stub that reads `cntvct_el0` and rescales it.
- **JIT:** a direct intrinsic.

PAuth, LSE, RCpc and 4K pages are free on HVF. A JIT backend for x86-64 (Steam Deck) still has
to implement them, at least as identity PAC like Eden-NX2.
