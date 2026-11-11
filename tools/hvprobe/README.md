# hvprobe

A small Rust program that checks whether Apple's Hypervisor.framework (HVF) can run Switch 2
guest code natively. It creates a VM, builds 4 KiB stage-1 page tables, installs an EL1 vector
table, and runs hand-written guest tests at EL1 and EL0. The guest code is in `src/guest.rs`;
the Hypervisor.framework bindings are in `src/hv.rs`.

```sh
./run.sh   # cargo build --release, ad-hoc sign with the hypervisor entitlement, run
```

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

What this doesn't cover yet: multiple vCPUs running at once, VM-exit cost under contention,
`hv_vcpus_exit` for preempting a vCPU, stage-2 remapping cost (lazy backing of sparse regions),
and exception-level details of a real HLE kernel (TPIDRRO_EL0, FP/SIMD state, EL0 data aborts
for fastmem-style tracking).

## What it means across backends

Counter reads need load-time patching under HVF. The same patch works under KVM on ARM64 Linux,
where physical counter access may also trap depending on the host, and it's harmless under an
x86-64 JIT, which can read its own clock directly. So the load-time **code patcher** belongs in
the shared loader, not in one backend. Each backend supplies its own stub:

- **HVF / KVM:** a stub that reads `cntvct_el0` and rescales it.
- **JIT:** a direct intrinsic.

PAuth, LSE, RCpc and 4K pages are free on HVF. A JIT backend for x86-64 (Steam Deck) still has
to implement them, at least as identity PAC like Eden-NX2.
