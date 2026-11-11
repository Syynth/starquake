# Port map: what starquake takes from Eden and oboromi

Layer-by-layer comparison of the emulators examined so far, and what to port from each.
Sources: Eden master (2026-10-29), Eden-NX2 (`Takoroka/eden-nx2` @ `50fa86bbb`), Eden PR #4588
(`ponting/eden@ns2-support` @ `e5fb3b97c`), oboromi @ `c97664a`. Lichii and Eglantine are Eden
forks with the same architecture as the Eden-NX2 column; their source wasn't examined.

**How to read the verdicts:**
- **Port:** translate the logic into Rust, keeping behaviour. Credit the source in the file header.
- **Reference:** read it for protocol facts and edge cases, then write our own design.
- **New:** nothing suitable exists.
- **Skip:** not needed.

## Summary

| Layer | Eden mainline | + Switch 2 work (NX2 / PR #4588) | oboromi | starquake crate | Verdict |
|---|---|---|---|---|---|
| CPU execution | dynarmic JIT (x86-64, arm64 hosts) and **NCE** (native on arm64 hosts, `core/arm/nce`) | dynarmic decodes ARMv8.1–8.5; PAC as identity (NX2) or QARMA (PR) | Unicorn, 8 cores, ARMv8.0 | `cpu` | **New** (HVF/KVM/JIT backends). **Port** NCE's `patcher.cpp` logic into `loader`. |
| Guest memory | `core/memory.cpp` page table + fastmem (`common/host_memory`), `device_memory_manager` for the SMMU | 12 GB layout, 35-bit device space, sparse regions | Flat 12 GiB `Vec` at VA 0 | `memory` | **Reference.** HVF stage-2 is a different mechanism. |
| Guest threads | `cpu_manager`: one host thread per core, guest threads are **fibers** (`common/fiber`) | 6 cores (PR), core folding (NX2, abandoned) | `run_all` single-steps each core | `kernel` + `cpu` | **New.** On HVF a guest context switch is a register swap on the vCPU; no fibers needed. |
| Kernel | `hle/kernel`: 183 files modelled on Horizon's K-objects, `svc/` (37 files) | `svcGetInfo` 29–33, `svc 0x80`, scheduler-lock spin | none | `kernel` | **Port** semantics (`KScheduler`, `KAddressArbiter`, `KPageTable`, handle table, SVC behaviour). Redesign the threading glue. |
| IPC | `hle_ipc` + `server_manager` (service threads) | TIPC reply fix (row 22) | 8-byte header parse | `ipc` | **Port** the framing rules; **Reference** for `server_manager`. Handle requests inline on the vCPU thread first. |
| Services | 74 service dirs (`sm`, `fs`, `am`, `hid`, `audio`, `nvdrv`, `nvnflinger`, …) | gsv, syncpt, gpu VM, vi2, codec sessions; `am`/`hid`/`aud:u` additions; PR #4588 `graphics_supervisor_protocol.h` | 160 empty structs | `services`, `gsv` | **Port** service by service. This is the bulk of the reusable logic. `gsv`: port NX2/PR behaviour, marked as reverse-engineered. |
| Loader / file system | `loader/`, `file_sys/` (NCA, NSO, NRO, RomFS, NPDM, patches), `crypto/` | Switch 2 ACID (0x300), loose ExeFS loading | mmap wrapper | `loader` | **Port.** Keys always user-provided. |
| GPU command processing | `gpu_thread` (one thread), `dma_pusher`, `puller`, `engines/` (Maxwell3D, KeplerCompute, MaxwellDMA, Fermi2D, KeplerMemory) | Ampere classes aliased to Maxwell engines; acquire suspend/resume, split methods, QMD 3.0, release slots; **front/back thread split** (NX2) | none | `gpu` | **Port** `dma_pusher`/`puller` with NX2's fixes, and register layouts. **Reference** engines: model Ampere classes natively, not by aliasing. Adopt the front/back split from day one. |
| Macros (MME) | `macro.cpp`: Fermi MME interpreter + x86-64 JIT | Turing MME64 interpreter, memo (97% replayed), x86-64 JIT (NX2) | none | `gpu` | **Port** the Turing interpreter and memo. JIT later, written for arm64 as well. |
| GPU memory | `memory_manager` (GPU MMU), `nvhost_as_gpu` | GPU VA == CPU VA via gpu sub-session 101/211; sparse 214/215 | none | `gpu` + `gsv` | **Reference.** The NVN2 model is new; port the protocol details. |
| Caches | `buffer_cache`, `texture_cache`, `query_cache`, `shader_cache` | collector thresholds, deferred inline invalidation, empty-fence skip, ASTC handling | none | `gpu` | **Reference.** These are tied to Eden's Vulkan renderer. Redesign for unified memory and Metal; keep NX2's perf lessons. |
| Shaders | `frontend/maxwell` → **IR** + `ir_opt` (SSA, structured control flow) → `backend/spirv` (Sirit), `glsl`, `glasm` | NX2: SM86 → lower to Maxwell SASS. PR: native Ampere frontend → IR (opt-in) | SM86 → hand-written SPIR-V, 5 of 201 ops | `shader` | **Port** Eden's IR and optimisation passes. Base the SM86 frontend on PR #4588's native frontend and NAK encodings. Use `naga` as the backend (SPIR-V + MSL). |
| Rendering | `renderer_vulkan` (+ OpenGL, null); macOS via MoltenVK | Vulkan fixes, BC re-encode compute shader | ash device only | `render` | **Reference** for Vulkan; **New** for Metal. |
| Video decode | `host1x/` (nvdec, vic, ffmpeg) | `nn::codec::video` via gsv, VP9/AV1/H.264 | none | `gsv` + host | **Port** the protocol; use FFmpeg through a crate behind a feature. |
| Audio / input | `audio_core`, `hid_core`, `input_common` | `aud:u` entry point, `REv?` revision tags | none | `services` + host | **Port** `audio_core` renderer logic and `hid_core` shared memory layouts. |
| Frontend | Qt (`yuzu/`), SDL (`yuzu_cmd`), Android | eden-cli SDL3 lifetime fixes | Qt 6 via Corrosion | `apps/` | **New.** CLI first. |

## Three structural differences to keep in mind while porting

1. **Execution model.** Eden runs guest threads as fibers on four host core threads, because
   dynarmic and NCE run guest code inside the host process and HLE code may block. Under HVF,
   guest state lives in the vCPU. A guest context switch saves and restores `GuestRegs` (and PAuth
   keys), and a blocking SVC means "pick another guest thread", not "park a host fiber".
   Port Eden's scheduler *decisions*, not its fiber plumbing.
2. **Where IPC is handled.** Eden dispatches service requests to service threads through
   `server_manager`. Starquake handles them inline on the calling vCPU thread at first: fewer
   hops, deterministic traces. It moves only long-running services to their own threads.
3. **GPU modelling.** Both Switch 2 efforts run Ampere classes on Maxwell engines and lower SM86
   to Maxwell SASS to reuse Eden's code. Starquake models Ampere natively, so port data and
   behaviour, not the aliasing layer.

## An option this raises: NCE on Linux arm64

Eden's NCE runs guest code directly in the host process on arm64. It patches SVCs and counter
reads (`core/arm/nce/patcher.cpp`) and handles faults with signal handlers and an interpreter
(`interpreter_visitor.cpp`). On **Linux arm64 with 4 KiB pages** that's a possible alternative to
KVM, with no `/dev/kvm` access needed. macOS is a poor fit for NCE (16 KiB pages, W^X rules), which
is why HVF is the Mac backend. Either way, NCE's patcher is the most directly portable piece of
CPU code.

## Porting rules

- Port by behaviour, in small reviewable pieces, each with a test that pins the behaviour, ideally
  a trace or fixture taken from the source implementation.
- File header: `// Ported from Eden <path> @ <commit> (GPL-3.0-or-later)`, or from oboromi,
  rapid-arm-emu or NAK with their own licence.
- Don't port Eden's workarounds for Eden's own bugs. The NX2 notes flag several, like the TIPC
  reply offset in row 22.
