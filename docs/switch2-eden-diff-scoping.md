# What Switch 2 support costs: Eden vs Eden-NX2 (and PR #4588)

Scoping notes for a Rust Switch 2 emulator on Apple Silicon. Written 2026-11-11 from the source
of the two Eden-based Switch 2 efforts.

## TL;DR

- **Two independent attempts exist, and both are Eden plus a large patch.** Eden-NX2 (Takoroka) adds
  ~16k lines over 172 commits. PR #4588 (ponting) adds ~37k lines in one squashed commit, 9k of
  which are tests. Both share an ancestor: a `changes.patch` that seeded the Ampere-to-Maxwell
  shader lowering.
- **Booting is cheap; the GPU is the project.** Of the 149 blockers in Eden-NX2's log, rows 1–30
  are CPU, loader, kernel and service work, and most of them were solved with stubs or by echoing
  values back. Rows 31–149 are almost all GPU: NVN2's memory model, the command stream, macros,
  shaders, textures, presentation and performance.
- **Neither attempt emulates a Switch 2 GPU.** Both run Ampere classes on Eden's Maxwell engines
  and lower SM86 shaders to Maxwell SASS so Eden's existing recompiler can take them. PR #4588 also
  has an opt-in native Ampere frontend into Eden's IR. A from-scratch design should go native from
  the start.
- **Apple Silicon removes a lot of their pain:** no ARMv8.1–8.5 decoder work (native execution), no
  ASTC transcoding (Apple GPUs decode ASTC in hardware), and no buffer read-back (unified memory).
  That's roughly 25 of the 149 rows.
- **New risks are specific to the Mac:** the guest counter frequency (31.25 MHz, read about 17M
  times a second), PAuth and MTE under Hypervisor.framework, 4K guest pages on 16K hosts, and
  sparse memory for the 20 GiB+ reservations games make.

## What was compared

| | Base | Branch tip | Size |
|---|---|---|---|
| Eden mainline | `eden-emu/eden` master | — | yuzu-derived, GPLv3 |
| Eden-NX2 | `26a03d247` (2026-10-26) | `Takoroka/eden-nx2` branch `eden-nx2` @ `50fa86bbb` | 208 files, +16,248 / −410, 172 commits |
| PR #4588 | `a2aa4d97e` (2026-10-29) | `ponting/eden` branch `ns2-support` @ `e5fb3b97c` | 308 files, +37,669 / −426, 1 squashed commit |

Where the changes landed (share of changed lines):

| Area | Eden-NX2 | PR #4588 |
|---|---|---|
| `video_core/` (GPU engines, Vulkan, caches) | 50% | 10% |
| `shader_recompiler/` (Ampere frontend and lowering) | 15% | 32% |
| `core/hle/service/` (gsv, nvdrv, am, hid, ...) | 19% | 20% |
| `dynarmic/` (CPU decoder) | 2% | 3% |
| tests | ~0 | 30% |
| tools / notes | 4% (`tools/nx2/*.py`, `NX2_NOTES.md`) | — |

Eden-NX2 spent its lines on performance and the GPU path; PR #4588 spent them on shader coverage
and tests. Eden-NX2's `NX2_NOTES.md` is the most useful single artifact: a table of 149 blockers
in the order they were hit, each with its cause and fix. Read it first.

Status as of the tips: Eden-NX2 reaches gameplay in Mario Kart World at ~17–22 ms frames on light
courses, with mesh-shader geometry missing. Splatoon Raiders plays its intro correctly.
PR #4588 is a draft. An Eden maintainer reviewing it estimated that about 40% of the code is
useful, and asked for a joint roadmap across the forks.

## What changes from Switch 1 to Switch 2, as an emulator sees it

### CPU and ISA
- **Instructions:** Games use ARMv8.1–8.5 heavily. Per title: ~115–135k `RETAA/RETAB`
  (pointer-authenticated returns), 37–100k `LDAPR` (RCpc loads), 25–32k LSE atomics, 6–9k `CAS`,
  2–4k `SWP`, a few `CASP`/`PACGA`/`XPAC`, and one MTE `STG`. Stock dynarmic decoded none of them.
  - Eden-NX2 implements atomics as an exclusive load/op/store loop, treats PAC as identity, and
    treats MTE tags as 0.
  - PR #4588 adds QARMA and per-process `enable_pauth_returns`.
- **Cores:** Six application cores (NPDM core mask `0x3f`), priorities 28–59, and system modules
  on virtual cores 6–7.
  - Eden-NX2 first folded cores 4 and 5 onto 0 and 1. That changed which heap job threads
    allocated from and caused memory-shortage crashes (rows 77–79), so it moved to six real cores.
  - PR #4588 sets `NUM_CPU_CORES = 6` and maps virtual cores 6 and 7 to the system core.
- **Counter:** `CNTFRQ` is **31.25 MHz** (Switch 1: 19.2 MHz). PR #4588 scales the counter for
  processes that use the Switch 2 ABI. Mario Kart World reads `CNTPCT` about **17 million times a
  second** while waiting on the GPU (row 127).

### Process, memory, kernel
- **NPDM:** The ACID has a 0x300-byte signature/key area (Switch 1: 0x200). Both branches probe
  for the ACID magic at +0x200, then at +0x300.
- **Address space:** 39-bit. The heap and alias regions scale with physical memory: 8–32 GiB heap,
  alias up to 8× that. Run with the 12 GB memory layout.
- **Memory budget:** Games plan all memory reported by `svcGetInfo` 21. Mario Kart World sets a
  3.125 GiB heap and puts a 6.33 GiB VAMM-backed root heap in the alias region. Every crash in
  rows 75–80 was memory accounting: GPU-only regions taken from the application pool while still
  reported as free.
- **New `svcGetInfo` ids:** 29 and 30 (remote region address and size), 31 and 32 (remote memory
  used and total), 33 (mapping alignment, page granule 0x1000).
- **New SVC:** `svc 0x80`, `SetGpuMemoryAttribute`.
- **Job scheduling:** The job system issues about 1,400 SVCs per frame, mostly `WaitForAddress`
  (0x34) and `SignalToAddress` (0x35) pairs. Eden's global scheduler lock is a bottleneck: three
  of six cores were seen blocked on it (row 109, still open).

### Services (HLE)
- **`am`:** `appletOE` cmd 10 is the new OpenApplicationProxy. `IApplicationProxy` gains cmd 60
  (graphics scale) and cmd 40. `IApplicationFunctions` gains 320 and 321.
  `ISelfController` 2000 returns a raw session to the graphics server.
- **`gsv` (graphics supervisor):** The new hub. Over TIPC it hands out sessions for:
  - syncpt (202/204, with a shared page layout and armed-wait bits),
  - GPU (497),
  - vi2 display and present (305; present is 229),
  - video codecs (902/906/1002),
  - image and native-memory management (111/112/114/423/429/430/435/1510/911/961/991).

  This is the biggest new protocol surface. `nx2_gsv.h` is 2,068 lines, and PR #4588's
  `graphics_supervisor.cpp` is 2,711 lines plus a protocol header.
- **GPU sub-sessions:** These implement NVN2's virtual-machine memory model:
  - 101 `CreateGpuAccessibleMemory`
  - 102 destroy
  - 211 `NormalMapGpuMemory4K`
  - 212 `NormalModifyMemoryKind4K`
  - 213 unmap
  - 214 `CreateSparseRegion`
  - 215 `DestroySparseRegion`
  - 201/202 VA partitions

  Jobs complete by signalling slot events.
- **`nvsrv` and NvRm:** This is the old nvdrv protocol with new expectations:
  - chip id `0x17E`,
  - a 0xD0-byte characteristics block,
  - TSG and 64 subcontexts,
  - flag bit 34 `SUPPORT_MAPPING_MODIFY`, without which compressible textures silently get GPU
    address 0 (row 149),
  - a read-only syncpoint shim page.
- **Other:** `aud:u` (new audio entry backed by audren, with `REv?` revision tags), `pl:u`
  2000/2010 (fonts), and new hid, ngc, chat and camera services (PR #4588).
- **TIPC bug in Eden:** `ResponseBuilder::Push(Result)` wrote the 2-word CMIF result, so every TIPC
  reply carrying data was off by one word (row 22). Get TIPC framing right from day one.

### GPU memory model
- **No more GPU address-space ioctls:** Switch 2 never maps through `nvhost-as-gpu`. Memory
  registered with gpu sub 101 is used at **GPU VA == CPU VA** (row 30).
- **GPU-only memory lives in sparse regions:** For example, six 300 MB regions at `0x81_0000_0000`,
  or two 5.7 GiB regions for Mario Kart World. The graphics server backs them, and render targets
  live there (rows 57, 80, 81, 85).
- **SMMU space:** Physical needs 34–35 bits; Eden-NX2 ended at a 64 GiB device address space.

### GPU command stream
- **Classes:** Ampere `C797` (3D), `C7C0` (compute), `C7B5` (DMA), plus class 0. NVN2 uses
  subchannel 2 (inline-to-memory) without binding it.
- **Gated command buffers:** Each command buffer is gated by a SEMAPHORED acquire that the CPU
  releases after writing it. The GPU must suspend and resume at the exact word (row 50).
- **Split methods:** A method's data can end mid-GPFIFO-entry and continue after a call to another
  buffer. Several of these can be pending at once (rows 51, 56, 73).
- **Queue-to-queue sync:** QMD release slots 24, 28 and 52 handle it. A bit-19 release is a
  max-release, and releases must latch the waiters they pass (rows 60, 62, 70, 84).
- **Turing MME64 macros:** NVN2 drives 3D and compute state and draws through these, about
  **1 million calls a second, 7 instructions each, 96% exact repeats**.
  - MME DMA reads feed the load FIFO.
  - Data RAM must be **per channel** (row 84).
  - Eden-NX2: interpreter, then a memo of pure calls (97% replayed), then an x86-64 JIT.
- **Compute:** Dispatched through QMD 3.0.

### Shaders
- **ISA:** SM86 SASS (128-bit instructions).
- **Missing pieces in the lowering:**
  - mesh and task programs (`ISBEWR`/`ISBERD`, used for high-detail LODs in Mario Kart World),
  - tensor-core HMMA/LDSM networks (a DLSS-style upscaler, replaced with a bilinear blit),
  - `TMML`,
  - 64-bit carry chains.
- **Semantics that changed:** `MUFU.SIN/COS` take turns, not radians (row 95). Binding the console's
  blit constant bank 24 is the system's job (row 65).
- **Validation:** `nvdisasm`, one instruction at a time (row 88). PR #4588 bases its encodings on
  **Mesa NAK's `sm70_encode.rs`, which is Rust and MIT-licensed.**

### Textures, video, presentation
- **Swizzle:** Textures use a Switch 2-specific GOB swizzle, including in ASTC (row 69).
- **Huge textures:** Virtual textures reach 8192×8192×68 ASTC 8×8. That's 1.5 GB in guest
  memory, or 23 GB once Eden decodes it to RGBA8 (rows 87, 94).
- **ASTC dominates VRAM:** About 700 ASTC textures cost ~4.5 GB as RGBA8. Rows 99, 100, 103, 107
  and 114 are all about transcoding or evicting them.
- **Video:** `nn::codec::video` is served through gsv, with VP9, AV1 and H.264 decoded via FFmpeg.
  It exposes NativeSyncHandles on a syncpoint the service allocates.
- **Presentation:** vi2 229 presents.
  - The game paces itself on its reply, so returning early lets the CPU run a full ring ahead
    (row 55).
  - vi2 events 31 and 32 must tick at 60 Hz, or Mario Kart World hangs before its first frame
    (row 82).

## Performance lessons that should shape the design

Eden-NX2's measurements (rows 98–145) point at the GPU thread, not the host GPU. The host GPU ran
20–40% busy while one emulator thread did everything.

1. **Split the GPU front from the back.** A front thread parses commands, runs macros and drops
   same-value writes. A back thread runs engine methods and records draws. They talk through a
   lock-free ring, and only acquires wait. Result: **1.46× draw throughput**, frames 17% shorter
   (rows 142–145). Rust's ownership model makes this easier to get right than it was in C++,
   where the bugs it found were cross-thread scheduler use.
2. **Memoize macros before JIT-compiling them.** A memo of pure calls hit 97%. The JIT saved only
   another ~15%.
3. **Drop same-value register writes.** About 60% of 6M writes a second were no-ops.
4. **Per-thread memory translation caches** with a generation counter; a shared one crashed
   (row 125).
5. **Skip empty fences, batch fence signalling, and don't download GPU-written buffers at every
   fence.** These three cut 250–330 MB/s of copies (rows 123, 124, 131).
6. **Cache rejected shaders negatively,** or you re-translate and re-log the same failure on every
   draw (rows 90, 112).
7. **Translate shaders in linear time;** a quadratic BFS took 3.4 s on one shader (row 121).

## What it means for a Rust design on Apple Silicon

### Problems that go away
| Their problem | Why it goes away |
|---|---|
| Decoding ARMv8.1–8.5 (row 3, ~200k instructions per title) | Guest code runs natively; M-series cores implement LSE, RCpc and PAuth. |
| Atomics emulated with exclusive-pair loops | Native. |
| ASTC transcoding to BC and VRAM blow-up (rows 94, 99, 100, 103, 107) | Apple GPUs sample ASTC natively. You still need to deswizzle GOB layouts, but not decode. |
| Copying GPU-written buffers back to the CPU (rows 115, 131) | Unified memory: `MTLStorageModeShared` buffers can alias guest memory directly. |
| Mesh and task programs with no pipeline (rows 89, 97) | Metal 3 has object and mesh shaders on Apple7+ (M1 and later). It's a different model from NV's, but a real target. |
| Huge virtual textures (row 87) | Metal sparse textures and placement sparse heaps. |
| x86-specific perf work (TSC frequency, PAUSE in counter loops) | Not applicable. |

### Problems that remain
- **The gsv, NvRm and NVN2 protocol surface.** It's large, mostly reverse-engineered in these two
  branches, and largely guessed (Eden-NX2's "next steps" #3 is "replace guesses with real
  protocol knowledge").
- **MME64:** you'll need an interpreter plus a memo, and an AArch64 JIT later (`dynasmrt`).
- **The SM86 shader translator.** This is the single largest piece of work. See the architecture
  section.
- **Semaphores, syncpoints and QMD release semantics.** These are subtle, and the notes are the
  best spec available.
- **Memory accounting** that matches what games plan for.
- **The tensor-core upscaler.** Start with the bilinear stand-in. Metal 4's tensor APIs may make a
  real implementation possible later (unverified).

### Risks specific to the Mac: measured on an M2, macOS 27.0.1 (see `tools/hvprobe/README.md`)
1. **The counter is the real problem, and patching solves it.** HVF guests see a 24 MHz counter
   and HVF can't scale it. Worse, **`CNTPCT_EL0` reads always exit to the host, at about 760 ns
   each**. At Mario Kart World's 17M reads a second that's about 13 host cores.
   - Trapping into the guest's own EL1 handler instead costs ~26 ns, about 45% of a core.
   - A load-time patch to a rescaling stub costs **~0.9 ns, about 1.5% of a core**.

   So a code patcher in the shared loader is mandatory, the way yuzu's NCE patcher works on
   Android. Keep the guest-EL1 trap as a safety net for sites the patcher misses.
2. **PAuth works natively. ✅** The guest sees PAuth2 + FPAC + PACGA. Keys set from the host apply,
   PAC values match across vCPUs and across E- and P-cores, and `pacibsp`/`retab` run unmodified.
3. **MTE `STG` raises UNDEFINED to guest EL1, and skipping it works. ✅**
4. **4 KiB guest pages on a 16 KiB host work. ✅** Only stage-2 needs 16 KiB alignment.
5. **SVC round trip ~0.76 µs** (EL0 → guest EL1 → host → back): ~1.1 ms per frame at 1,400
   SVCs. Fine to start with.
6. **The guest physical space is 36 bits (64 GiB) on M2**, which fits the 12 GB layout.
   Sparse GPU reservations (20 GiB+) are virtual-address bookkeeping, but back them lazily, and
   keep accounting honest.

**Across backends:** the code patcher and its counter stub work under HVF and KVM, and are
harmless under an x86-64 JIT. On x86-64 (Steam Deck class), the JIT has to provide PAuth, LSE
and RCpc itself, which HVF gives for free.

## Proposed architecture (first cut)

```
crates/
  hos-loader     NSO/NRO/NPDM (both ACID layouts), ExeFS + RomFS, code patcher (cntpct, svc, pauth fallback)
  hos-cpu        CpuBackend trait: HVF (macOS arm64), KVM (Linux arm64), JIT (x86-64, Steam Deck class);
                 6 app vCPUs + system, guest EL1 stub vectors, stage-2 manager (per decision log, Linux is first-class)
  hos-kernel     HLE kernel: handles, threads, sync (WaitForAddress fast path), memory regions, svcGetInfo
  hos-ipc        CMIF + TIPC framing (shared types, property tests), session/port plumbing
  hos-services   sm, fsp-srv, am/appletOE, hid, aud:u, pl:u, ... (start auto-stub + trace like EDEN_NX2_AUTO_STUB)
  nx2-gsv        gsv + syncpt + gpu VM sessions + vi2 + codec sessions
  nx2-gpu-front  GPFIFO/pushbuffer parser, semaphore acquire/suspend, split-method parking, MME64 (interp → memo → JIT)
  nx2-gpu-back   engine state (C797/C7C0/C7B5), QMD 3.0, draw/dispatch recording
  sm86           SASS decoder + IR (NAK as reference) → backend (naga module → MSL, or direct MSL)
  render-*       Vulkan (Linux) and Metal (macOS) backends behind one trait: pipelines, sparse resources,
                 GOB deswizzle compute, present. Vulkan needs ASTC transcoding on AMD (Steam Deck); Metal doesn't
  tools/         nso-syms, a64/sm86 coverage, gdbstub, ipc/svc trace viewers
```

A note on the shader backend: building a `naga::Module` and emitting MSL with naga's backend
keeps the shader pipeline in Rust and gives you WGSL/SPIR-V for debugging. Check naga's coverage
early: mesh shaders, subgroup ops, 64-bit atomics, bindless and sparse residency. If it's missing
pieces you need, generate MSL directly from your own IR.

### Milestones
1. **libnx homebrew hello-world.** No keys involved. Proves HVF vCPUs, the loader, SVCs and
   CMIF/TIPC.
2. **A Switch 2 ExeFS boots to the graphics init.** Equivalent to Eden-NX2 rows 4–29. Mostly
   NPDM, svcGetInfo 29–33, six cores, and the am/gsv/syncpt stubs with tracing. Expect weeks, not
   months.
3. **First NVN2 present.** Rows 30–57: the GPU memory model, gated command buffers, MME
   interpreter, minimal SM86 translator, and vi2 229 with proper pacing.
4. **Correct frames in one title.** Shader coverage, ASTC/GOB textures, QMD release semantics.
5. **Performance.** Front/back split, macro memo, then the JIT.

## Licensing and references

- **Eden, Eden-NX2 and PR #4588 are GPLv3.** Code translated from them makes your project GPLv3.
  Reading them to learn protocol facts (command ids, struct layouts, behaviour) is a different
  matter from copying expression. If you want a non-GPL project, keep a clean separation and
  document where each fact came from. This isn't legal advice.
- **Mesa NAK** (`src/nouveau/compiler/nak/`, Rust, MIT): SM70+ instruction encodings and an
  example of a SASS-level IR in Rust. It's the best reference for the shader translator, and it's
  license-compatible with anything.
- **oboromi** (`github.com/0xNikilite/oboromi`, Rust, GPLv3): an existing Switch 2 emulator
  foundation in Rust, targeting SM86 with an ARMv9 interpreter. It can't run games yet. Worth
  reading, or contributing to, before starting from zero.
- **`NX2_NOTES.md`** on `Takoroka/eden-nx2` and **`graphics_supervisor_protocol.h`** on
  `ponting/eden@ns2-support`: the closest thing to a protocol spec right now.
- **Keys, firmware and games** must come from the user's own hardware. Never ship or fetch them.

## Open questions

- Do you want GPLv3 (and freedom to port from Eden) or a permissive licence (clean-room only)?
- Start from scratch, or contribute an Apple Silicon/HVF backend to oboromi?
- Which game is the first target? Eden-NX2's notes suggest Splatoon Raiders (title 4) is the
  most forgiving; Mario Kart World needs mesh shaders and sparse textures.
- Is the hardware/firmware-dump path settled, so you have legally obtained test titles?
