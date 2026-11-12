# Review guide: the four graphics drafts

Status: a reading aid for the owner, not a spec. It compares four draft design PRs that were
written in parallel by agents who could not see each other's work. On 2026-11-12 the owner took
the six foundational decisions below, and the four drafts were amended to follow them and the
seam resolutions marked "resolved" here. Everything else in this guide is still a recommendation.

| Short name | Draft | PR | Issue |
|---|---|---|---|
| graphics-server | `docs/specs/graphics-server.md` on `design/graphics-server` | #109 | #11 |
| gpu-protocol | `docs/specs/gpu-protocol.md` on `design/gpu-protocol` | #116 | #12 |
| shader-pipeline | `docs/specs/shader-pipeline.md` on `design/shader-pipeline` | #110 | #13 |
| render-backend | `docs/specs/render-backend.md` on `design/render-backend` | #114 | #14 |

"Decision N" below is the numbered item in that draft's "Decisions for the owner". A slice id is
the draft's own (graphics-server 1–16, gpu-protocol G1–G21, shader-pipeline 1–21,
render-backend R0–R12). Every claim about a draft was checked against its text; where the texts
do not settle something, this guide says so.

## Decided so far

The owner's rulings of 2026-11-12, in `docs/decision-log.md`. Each overrides anything in the
drafts and in this guide; where this guide recommended otherwise (M1), the ruling wins.

1. **wgpu first, behind our own trait** (render-backend decision 1, option e). No wgpu type
   crosses into `gpu`. A direct Vulkan backend comes when Linux work or a title needs a stage wgpu
   lacks; a native Metal backend only if measurements at the performance milestone call for it.
2. **Shaders reach the backend as a naga module; naga is the only shader backend for now**
   (shader-pipeline decisions 2 and 3). The shader crate hands out a validated module plus its
   resource, input and output tables; it emits final SPIR-V or MSL only for a later direct
   backend that asks. `shader` and `render` do not depend on each other; all use wgpu's naga.
3. **The game sees GPU work as done only when the host GPU has finished it** (gpu-protocol
   decision 2, option b). Signals with no host work behind them at once, batched; record-time
   signalling only as a debugging switch.
4. **Presents pace the game through a buffer queue** (graphics-server decision 1). The present
   reply returns at once; the 60 Hz tick shows one frame and frees the buffer it replaces; the
   fence wait is a per-run fallback as a deferred reply.
5. **Syncpoints and the GPU address table live in `crates/gpu`; gsv decides, the GPU applies**
   (graphics-server decision 3, with M4 and M5). `SyncpointSink` is dropped; ordered unmaps
   with a barrier, "finished" covering the front thread too; `GpuMemory` kept with its production
   implementation in `crates/gpu`.
6. **Command lists as data; caches split by what they are keyed on** (render-backend decisions
   3 and 4).

## 1. How the four designs fit

- **Guest → graphics-server (`crates/gsv`):** IPC requests. gsv validates them and owns sessions,
  syncpoint allocation, GPU memory bookkeeping, channels and vi2 present pacing.
- **graphics-server → gpu-protocol (`crates/gpu`):** GPFIFO submissions, channel and address-space
  changes, and latched presents, through `GpuFront`. Back the other way: syncpoint increments and
  "done" reports, as `GpuCompletion`s pushed into gsv's completion queue.
- **gpu front → gpu back:** one ring of plain-data entries (register writes, releases, presents).
- **gpu back → shader-pipeline (`crates/shader`):** a program address plus answers about engine
  state; back comes a shader set (a naga module per stage), a pending ticket, or a rejection, with
  its resource table.
- **gpu back → render-backend (`crates/render`):** resource creation by description, and command
  lists (draws, copies, presents) as data. Back: fence values, readbacks, present feedback.
- **shader-pipeline → render-backend:** a naga module and its tables (decision 2). It passes
  through `gpu`, because `shader` and `render` must not depend on each other.
- **Kernel and memory → gpu:** guest memory reads and CPU-write tracking (a new seam, section 3).

## 2. Seam mismatches

Each item: what the drafts say, why it matters, the options, and a recommendation. Section names
are the drafts' headings.

### M1. What the shader pipeline hands the render backend

**Status: resolved by decision 2, against this guide's recommendation (a).** The shader crate
hands a validated naga module and its tables to the wgpu backend through `gpu`; it emits final
code itself only for a later direct backend that asks (shader-pipeline slice 22). No crate but
`shader` writes final code from guest shaders. The analysis below is kept for the record.

- **shader-pipeline** ("What the render backend (#14) gets"): finished host code,
  `HostCode::SpirV(Vec<u32>)` or `HostCode::Msl(String)`, emitted by the shader crate through
  naga. The binding convention is defined once in the shader crate, as data in `ResourceSlot`; the
  render backend "does not guess it". It wants only its lowering module to touch naga's API,
  because naga breaks every few months ("The backend through naga").
- **render-backend** ("Shader emission", "Shader pipeline (#13)"): a validated `naga::Module` plus
  `ModuleInfo`. The backend runs naga's SPIR-V or MSL writer itself, "because only it knows its
  binding model", and sets naga's bounds-check policy from `Caps`. `gpu` converts the resource list
  into the render crate's `ShaderInterface`.
- **Why it matters:** it decides which crates use naga's API, what the shader disk cache stores
  (final bytes or a naga module), who owns binding numbers, and which slices build emission
  (shader-pipeline 16–17 or render-backend R6).
- **Options:** (a) the shader crate emits; the render backend passes its binding and bounds-check
  needs in as data (`HostTarget`). (b) The shader crate stops at a validated `naga::Module`; the
  backend emits.
- **Recommended: (a).** It keeps guest-shader code on one side of naga's churn, lets the disk cache
  and the shader tests (MSL goldens, `spirv-val`) cover the exact code that runs, and the render
  draft already accepts SPIR-V and MSL (`ShaderSource::{Naga, SpirV, Msl}`). The cost: the binding
  map and the bounds-check policy become fields of `HostTarget`, and render-backend R6 builds
  pipelines from given code. The render crate still uses naga for its own WGSL helper shaders
  (render-backend, decision 12). If the owner picks wgpu (render-backend, decision 1, option e),
  both forms still work: wgpu takes a naga module or SPIR-V/MSL passthrough.

### M2. How presents pace the guest

**Status: resolved by decision 4.** The render and gpu drafts now say back-pressure reaches the
guest through later fences and later buffer returns, with no held reply. The route gsv → gpu →
render stands; correcting `architecture.md` is still a proposal (section 3).

- **Route: agreed by three drafts, against the architecture doc.** graphics-server ("What this
  spec offers and assumes of the render backend"), gpu-protocol (ring entry `Present`, ordering
  rule 6) and render-backend ("Graphics server (#11)") all send a present gsv → gpu → render, so it
  is ordered after the draws that made the frame. `docs/architecture.md` draws gsv → render (its
  diagram and "How a frame flows", step 6). Fix the architecture doc (section 3).
- **Pacing: not agreed.** graphics-server builds a buffer queue: the vi2 229 reply comes back at
  once, a buffer returns only after a later frame replaced it on screen, and "the handler never
  waits" ("Present (229) and pacing"; its decision 1). render-backend assumes the other model:
  "the graphics server's present reply (which waits on them, row 55) holds the game" ("Command
  lists are data") and "the present reply still waits on the guest's fence" ("Compiling without
  stalling a frame"). gpu-protocol says "Present waiting with a bounded timeout is gsv's"
  ("Graphics server sessions (#11)"), which also reads as the fence-wait model.
- **Why it matters:** render-backend's story for how a slow host GPU slows the guest rests on a
  reply that graphics-server never holds.
- **Recommended:** take graphics-server decision 1 first (section 5). If the buffer queue stands,
  the render and gpu drafts change one paragraph each: back-pressure reaches the guest because
  fences pass later and buffers come back later, not because a reply is held.

### M3. Who says a presented buffer is free again

**Status: resolved by decision 4, as option (b)** (the tick "frees the buffer it replaces").
`PresentReleased` and the release ticket are gone from the drafts. The ordering inference below
is still unconfirmed, so the drafts mark it as a guess: render-backend slice R8 pins it with a
test, and graphics-server verification 12 says what changes if it fails.

- **graphics-server:** the render backend pushes `PresentReleased` into gsv's completion queue
  "once it no longer reads the image"; only then does the slot come back (steps 2–3 of the latch;
  `GpuFront::present(request, released: Ticket)`).
- **gpu-protocol:** `Gpu::present(request)` has no release ticket and is listed as called from
  vCPU threads; gsv calls it from the timer thread.
- **render-backend:** `PresentFeedback` (shown or dropped, host time) is "for statistics only;
  gsv must not let it change guest-visible behaviour" ("Pacing at 60 Hz"). Its trait has no
  `present` method; a present is `Command::Present` inside a command list.
- **Why it matters:** gsv's design needs a guest-visible signal that the render draft forbids,
  and no draft carries it across the gpu layer.
- **Options:** (a) as graphics-server: release on a render report, routed render → gpu → gsv,
  recorded as a `BufferReleased` note so replay still works; host timing then decides when a
  buffer returns. (b) Release the replaced frame's slot on the vsync tick that latches the next
  frame. This leans on an inference, not on any draft's text: the present blit and any later
  draw into the same image are encoded in order on render-backend's one host queue, with the
  barriers it inserts, so reuse cannot corrupt the frame being shown.
- **Recommended: (b), if the gpu and render authors confirm that ordering;** otherwise (a). (b)
  makes buffer return a pure function of the 60 Hz clock, matches render-backend decision 8, and
  removes `PresentReleased` and the release ticket from the interfaces.

### M4. Syncpoints: where the counters live, who writes them, and when

**Status: resolved by decisions 3 and 5, as recommended.** `Syncpoints` in `crates/gpu`,
incremented after host completion, guest events signalled only from gsv's timer-thread drain;
`submit` carries the wait fence; `SyncpointSink` dropped.

- **graphics-server** (decision 3; "What this spec offers and assumes of the GPU front/back"): a
  `Syncpoints` type of 192 atomic counters, with maxima and a watched bitmap, lives in
  `crates/gpu`. The GPU and the video worker increment it in place; a watched increment pushes
  `SyncpointChanged` into gsv's completion queue, which the timer thread drains in one recorded
  order. GPU shim reads and submission fence waits read it directly. gsv computes each
  submission's fence. It assumes increments happen "when the work before them has been executed
  by the back thread (that is, recorded for the host GPU), not when the host GPU has finished it".
- **gpu-protocol** ("Graphics server sessions (#11)", "Releases and completion"): "gsv owns
  syncpoints". The back thread calls a `SyncpointSink::increment` that gsv implements, and gsv
  "wakes its waiters through kernel handles" from that call. `Gpu::submit` returns the value the
  fence will reach. Increments are performed only after the host GPU has finished the work before
  them (its decision 2). Its `submit` has no parameter for the fence to wait on first (flag bit 0).
- **Why it matters:** three disagreements at once: the type's home, the thread that signals guest
  events, and what "reached" means. The last decides whether the guest can reuse memory the host
  GPU is still reading.
- **Recommended:** graphics-server's type in `crates/gpu`, written by the back thread (and the
  video worker), with kernel events signalled only from gsv's timer-thread drain, so their order
  is recorded (observability rule 3). gsv computes the fence (it owns allocation and maxima); the
  GPU performs the given number of increments; `submit` gains the wait fence. On timing, take
  gpu-protocol decision 2 (b), after host completion, and change graphics-server's assumption to
  match. Drop `SyncpointSink`.

### M5. GPU address translation and ordered unmaps

**Status: resolved by decision 5, as recommended**, including "finished" covering the front
thread. The shared interface (`GpuFront` with `update_vm` and `barrier`, `GpuCompletion`) is now
word for word the same in graphics-server and gpu-protocol. Applying it surfaced two open cases,
listed in gpu-protocol ("Ordered address-table updates and barriers"): a parked channel holds
later ordered updates of its process, and work queued after an update may still see the old
mapping until it applies.

- **graphics-server** (decision 3; "How mappings reach the GPU"): gsv publishes a versioned
  `GpuVm` table type that lives in `crates/gpu`; the GPU reads only that. Rejected option: "in gsv,
  behind traits the GPU calls" (a virtual call per translation). Unmaps, kind changes and destroys
  go to the GPU as ordered updates (`update_vm(space, update, done)`, `barrier(done)`); gsv
  signals the guest's job event only when the GPU reports no earlier work can still read through
  the old mapping (row 68).
- **gpu-protocol** ("Reading and writing guest memory"): a `GpuMemory` trait (read, write, raw view,
  `generation()`) defined in `crates/gpu` and **implemented by gsv**, which "bumps the generation
  on every change". There is no ordered-update or barrier call in its interface; its internal
  `Marker` ring entry does the same job inside the GPU.
- **Why it matters:** an immediate generation bump cannot provide the ordered unmap that row 68
  needs, and a generation number does not say which range changed, which cache invalidation
  needs (M7).
- **Recommended:** `GpuVm` in `crates/gpu`, maps written by gsv, ordered updates applied by the GPU.
  Keep `GpuMemory` as the GPU's access trait, but put its production implementation in `crates/gpu`
  over `GpuVm` plus `GuestMemory`; the capture-replay implementation stays as gpu-protocol has it.
  Add gsv's `update_vm` and `barrier` to the gpu interface, reported through a `Marker`. Open, in
  neither draft: "finished" must cover the front thread too, since it reads command words through
  the same table.

### M6. GPU-only memory (sparse regions): who backs it

**Status: resolved as recommended** (applied 2026-11-12). gpu-protocol has a "GPU-only memory"
section and slice G22; graphics-server says gsv keeps extents and accounting. graphics-server
decision 4 (lazy host backing at all) is still open.

- **graphics-server** ("GPU-only memory"; decision 4): backed lazily on the host, "the GPU side
  (#12, #14) allocates host storage"; gsv keeps the extents.
- **gpu-protocol** ("Reading and writing guest memory"): GPU-only regions are "sparse regions the
  graphics server backs", resolved by gsv's `GpuMemory`.
- **render-backend** ("Block-linear data"): a render target that only the host GPU uses "exists
  only as a host texture".
- **Why it matters:** each of the first two says the other allocates it, and no slice builds it.
- **Recommended:** `crates/gpu` owns the lazy store (only GPU threads touch it): render targets
  there stay host-only textures in the texture cache, other bytes go in a gpu-side sparse store.
  gsv keeps extents and accounting only. A slice is needed (section 4).

### M7. Write tracking and shader-cache invalidation

**Status: resolved as recommended** (applied). The kernel half of `WriteWatch` is still unowned
(section 4), and the execution-model change is still a proposal (section 3).

- **gpu-protocol** ("Learning of CPU writes"): a new `WriteWatch` type in `starquake-memory`. The back
  thread requests protection; the kernel applies it (with a TLB flush) at the next kernel entry,
  records faults as dirty pages and restores access; the back thread drains the dirty log before
  each draw, dispatch and copy.
- **shader-pipeline** ("Identity, caching and invalidation"): `ShaderCache::invalidate(range)` is
  "callable from any thread (fault handlers included)"; `gpu` calls it for GPU-side writes and
  unmaps and registers `tracked_ranges()` with write tracking. It assumes gsv tells `gpu` of
  unmaps (gpu worker 102, 213, 215).
- **graphics-server:** says nothing about write tracking.
- **Why it matters:** a fault handler is kernel code, and the kernel cannot call the shader crate
  (dependency direction). Unmap ranges have no carrier until M5 is settled.
- **Recommended:** one path. The back thread drains `WriteWatch` and calls `invalidate` for the
  shader cache as for its own caches. Unmaps arrive as M5's ordered updates, which carry the range.
  The "fault handlers included" line goes. The kernel half is unowned work (section 4).

### M8. Tiling: the layout crate and who detiles

**Status: resolved as recommended** (applied). gpu-protocol decision 5 (the crate itself) is
still open; it now notes that with `gpu` as the only production caller, option (a) would work
today.

- **gpu-protocol** ("Block-linear layout"; decision 5): a new leaf crate `starquake-layout`, ported
  from Mesa's `nil`, with three callers: the gpu back thread, the render backend (a GPU deswizzle
  must agree with it) and gsv (native image sizes). It also derives that the Switch 2 GOB is Mesa's
  `TuringColor2D`, checked by its first test.
- **render-backend** ("Block-linear data"): detiling happens on the `gpu` side before upload; the
  backend gets linear data. `starquake-render` "depends on no workspace crate". Its verification 5
  asks the same GOB question that gpu-protocol already answers.
- **graphics-server** ("What vi2 needs from the image"): "detiling ... is the render side's". Its
  slice 8 computes native image sizes with its own formula, ported from PR #4588
  (`CalculateNativeImageRequirements`), and never mentions a layout crate.
- **Why it matters:** gpu-protocol counts gsv and render as callers, but neither draft plans to
  call it, and graphics-server and render-backend disagree on who detiles.
- **Recommended:** the crate as gpu-protocol proposes, called by `gpu` for all detiling and
  write-back. Correct graphics-server's sentence: `gpu` detiles. gsv keeps its formula, because it
  reproduces the system module's answer (a protocol fact), and a test checks it against the layout
  crate's size for the same description. The render crate does not depend on the layout crate
  until GPU detiling is built (deferred in render-backend). Close render-backend verification 5
  with gpu-protocol slice G5.

### M9. How the GPU asks for a shader

**Status: applied as recommended** (gpu-protocol now uses `lookup`). The owner's decision behind
it, shader-pipeline decision 5, is still open.

- **gpu-protocol** ("Shader pipeline (#13)"): `translate(request, env)`, "called synchronously on
  the back thread"; `env` reads program words, constant-buffer words and texture types live during
  the call, and the reads become part of the cache key.
- **shader-pipeline** ("Translation inputs", "Compilation, threads and stutter", its decision 5):
  `lookup` never blocks and never translates; it returns `Ready`, `Pending(Ticket)` or
  `Rejected`. Translation runs on worker threads from explicit answers (`CbufWord`,
  `TextureKind`) that the back thread gives through `EngineState`. Live reads during translation
  are the option the draft rejects (row 121).
- **Why it matters:** the two cannot both be built. The content is close (both key on what was
  read), the threading is not.
- **Recommended:** shader-pipeline's form (its decision 5, option c). gpu-protocol's back implements
  `EngineState` and `ProgramMemory` and owns the policy for a pending ticket (M10). gpu-protocol
  slice G18's fake shader pipeline uses the `lookup` signature.

### M10. A draw whose shader or pipeline is not ready

**Status: structure applied as recommended** (one setting owned by `gpu`; translation waits on
the back thread, pipeline waits on the render thread). The default (B1) is still open.

All three drafts recommend **wait by default, skip as an option later**, so there is no conflict
of substance. They differ in detail:

- gpu-protocol (decision 9): wait to start, skip later; translation is synchronous anyway (M9).
- shader-pipeline (decision 6): wait by default, skip "once the persistent cache exists", which
  its decision 7 puts after milestone 3. The policy is the GPU crate's.
- render-backend (decision 5; "Compiling without stalling a frame"): a setting; wait "until
  milestone 4", revisited with the disk cache at milestone 5. Pipeline waits happen on the render
  thread, not the back thread.
- **Recommended:** one decision (section 5, B1): one setting owned by `gpu`, default wait.
  Translation waits block the back thread and pipeline waits the render thread; skip arrives with
  the persistent caches.

### M11. Texture component swizzles

**Status: resolved by decision 1.** The render backend reports `Caps::view_swizzle` (false under
wgpu); `gpu` folds each binding's swizzle into the shader variant key; shader-pipeline decision
16 carries the exception. The MoltenVK check below moved to render-backend's deferred direct
Vulkan backend.

- **render-backend** ("Textures, views, samplers, render targets"): the per-texture swizzle goes in
  the host view (Vulkan's view, Metal's texture swizzle), so it "costs no shader variants". Its
  wgpu table: wgpu has no view swizzle, so under wgpu it "must become a shader variant per
  binding", which it calls the one wgpu gap that bites at first frames.
- **shader-pipeline** ("Pipeline-state specialisation"): the variant key has no swizzle.
- **Agreed under render-backend decision 1, options b–d.** Under option e (wgpu first), the variant
  key gains a swizzle per texture binding, and shader-pipeline decision 16 ("types and structure
  only") would need an exception.
- **Gap:** no draft checks whether MoltenVK, the Mac bring-up path under option b, honours view
  swizzles. Add it to render-backend verification 1.

### M12. Geometry, tessellation, transform feedback, mesh

**Status: catch (1) resolved by decisions 1 and 2** (neither wgpu nor naga runs these stages,
so they wait for a direct Vulkan backend; shader-pipeline decision 13 is settled in effect).
**Catch (2), the mesh MSL fact, is still open**: shader-pipeline experiment E1 settles it.

- **Mostly consistent.** shader-pipeline defers geometry and tessellation (decision 13) and
  transform feedback; it keeps them representable (`GuestStage`, the SPH parser). render-backend
  keeps them in `GraphicsPipelineDesc`, skips and counts geometry draws until a compute emulation
  exists, and defers transform feedback. gpu-protocol defers transform feedback and lets
  `DrawDesc` carry mesh programs.
- **Two catches.** (1) render-backend counts the lack of these stages against wgpu and rates
  Vulkan "Full on Linux" (decision 1), but naga cannot emit them for Vulkan either (shader-pipeline
  gap table). Under both options, nothing reaches the host until shader-pipeline decision 3
  chooses a direct SPIR-V writer (option c) or an upstream change, so this argument does not
  separate options b and e for now. (2) A factual conflict on mesh shaders: shader-pipeline says
  naga 30.0.1's MSL writer emits `[[object]]`/`[[mesh]]`; render-backend's wgpu table says "naga
  can't emit them there". shader-pipeline's experiment E1 (a mesh entry point written as MSL)
  settles it.

### M13. What shaders are specialised on: one key, defined twice

**Status: resolved as recommended** (applied).

- **render-backend** ("Fixed-function state the host lacks"): a pure `emulation(&Caps, &state) ->
  EmulationFlags` in the render crate decides alpha test, depth-range rewrite, attribute fetch
  beyond 16, typed outputs and primitive-restart handling. The shader pipeline specialises on the
  flags, and they are part of the pipeline key.
- **shader-pipeline** ("Pipeline-state specialisation"): a `VariantKey` built by `gpu` from engine
  state: attribute and target component types, dual-source blending, interface matching, alpha
  test, depth range "where the host cannot express them as state", compute sizes. No attribute
  fetch from a buffer and no primitive restart.
- **Why it matters:** the shader crate cannot see render's `EmulationFlags` (no dependency), and two
  lists will drift.
- **Recommended:** render's `emulation()` decides what the host lacks (it has `Caps`); `gpu` copies
  the result into the shader crate's `VariantKey`, whose list grows by render's extra items. Numbers
  stay in driver constants (shader-pipeline decision 16).

### M14. The shape of render commands and completion

**Status: types resolved by decision 6** (render-backend's `CommandList`, `Command`,
`FenceValue`, `completed()`; gpu-protocol's sketch replaced). **The back thread's wake on
completion is still open** in both drafts; gpu-protocol slice G15 waits on it.

- **gpu-protocol** ("What the back hands the render backend", "Render backend (#14)"): sketches
  `BeginPass`/`EndPass`/`Barrier` commands, a `present(PresentFrame)` trait method, and "a
  completion signal per submission that the back can wait on together with the ring". It says
  #14 owns the final types.
- **render-backend** ("Commands", "The trait"): passes and barriers are "the backend's business"
  (`SetRenderTargets`, backend-opened passes); present is `Command::Present`; completion is
  `completed()` and `wait(value, timeout)`.
- **Recommended:** render-backend's types, as gpu-protocol already defers to them. Still open in both:
  how the back thread sleeps on "ring has entries **or** a submission completed" with only
  `completed()`/`wait()`. gpu-protocol slice G15 needs an answer. One way is a wake hook the back
  thread installs; neither draft has one.

### M15. Three capture formats, one command name

**Status: resolved as recommended** (applied: render-backend R11 is `starquake render-replay`;
shader answers are in `SQGC`). Recording the new commands in the observability spec is still a
proposal.

- gpu-protocol ("Capture and replay"; decision 8): format `SQGC` of GPU-thread inputs, replayed by
  `starquake gpu-replay <capture> --until present=N --dump ...`.
- render-backend slice R11: host-level capture of command lists and staging bytes, replayed by
  `starquake gpu-replay <capture> --png <dir>`.
- shader-pipeline slice 20: a local shader capture (program bytes plus answers),
  `cargo xtask shader-replay`. It assumes gpu-protocol's capture records shader requests with
  their answers. gpu-protocol's list of what a capture holds does not mention them.
- **Recommended:** keep all three, since they serve different layers, but give R11 its own name (for
  example `render-replay`). Add the shader answers to the `SQGC` contents. Record the new
  commands in the observability spec's headless contract (section 3).

### M16. Who owns the compile worker threads

**Status: resolved as recommended** (applied: `core` creates one pool; render-backend R12 wires
it).

- shader-pipeline ("Compilation, threads and stutter"): the shader crate owns no threads; the caller
  hands it an executor "so translation and the render backend's pipeline compilation can share
  one pool".
- render-backend ("Where the backend sits"): compile workers are owned by the backend; the `Backend`
  trait exposes no executor.
- **Recommended:** `core` creates one pool and hands it to both. That needs no dependency between
  `shader` and `render` and sizes threads in one place. A small change to render-backend.

### Candidates checked and dismissed

- **Constant bank 24 (row 65).** Not a mismatch: graphics-server, gpu-protocol and shader-pipeline
  all leave supplying it to `gpu`. The fact itself is open (gpu-protocol verification 6).
- **Semaphore mirror page (row 35).** Consistent: gsv lists two-page registrations, `gpu` decides
  (graphics-server "The semaphore mirror page"; gpu-protocol verification 7). Neither interface
  sketch carries the list yet.
- **Negative caching of rejected shaders and failed pipelines.** Consistent in all three.

## 3. Changes to approved specs that the drafts imply

These need the owner's approval in their own right, separately from the drafts. The six
decisions do not change approved specs by themselves; rows touched by them are updated below.

| Spec | Change | Implied by |
|---|---|---|
| execution-model, "Write tracking for the GPU caches" | The kernel no longer "notifies the cache". A `WriteWatch` in `starquake-memory` is requested by the GPU back thread, applied by the kernel at the next kernel entry on any core, and drained as a dirty log | gpu-protocol ("Approved specs this design leans on") |
| kernel, "Address space" and port plan | Device-share lock and unlock over a range, and a range query of the GPU-accessible attribute that `svc 0x80` sets, callable by gsv | graphics-server ("What this spec needs from the kernel") |
| kernel | A hook on events a service created, run after the kernel lock is released | graphics-server (decision 9) |
| kernel, "Address space" (pool accounting) | GPU-only memory is never charged to the title's pools. Probably consistent with "what was actually reserved", but the kernel spec names memory the graphics server takes; confirm | graphics-server (decision 4) |
| services, "The service manager" | Registry entries reachable only from other services (gsv's root for `am` 2000) | graphics-server ("The root") |
| services, `Context` | The "narrow kernel interface" grows: the event hook, the address-space queries, the device-share lock, transfer memory | graphics-server |
| observability, artifacts table | A `gpu.txt` flight-recorder artifact | gpu-protocol ("Observability") |
| observability, `summary.json` | GPU counters; shader rejections counted once per run; skipped draws and failed pipelines counted | gpu-protocol, shader-pipeline, render-backend |
| observability, headless contract | New commands `starquake gpu-replay` and `starquake render-replay` (M15, resolved), and a `--window` flag | gpu-protocol, render-backend (R11, decision 11) |
| observability, "Later" | The GPU capture format and the recording backend now have specs (anticipated there, so an update, not a reversal) | gpu-protocol, render-backend |
| observability, `ServiceNote` | New note kinds (`VsyncTick`, `FrameLatched`, and others). Within the existing event shape; listed only for completeness | graphics-server ("What is recorded") |
| architecture.md, diagram and "How a frame flows" step 6 | Present goes gsv → gpu → render, not gsv → render | graphics-server, gpu-protocol, render-backend |
| architecture.md and `CLAUDE.md` dependency rules | New leaf crate `starquake-layout`; `starquake-render` split into a trait crate plus `starquake-render-wgpu` (working name; later `-vulkan`, and `-metal` only if measured to be needed), chosen by `core`; possibly `starquake-sm86` | gpu-protocol decision 5; render-backend decision 10 (implied by decision 2's wording); shader-pipeline decision 14 (recommends not) |
| architecture.md, platform matrix | Every platform starts on wgpu (Metal on the Mac, nothing to install; Vulkan on Linux); direct Vulkan later, native Metal only if measured; copies, no aliasing, until a direct backend | Decision 1; render-backend decision 7 (moot for now) |
| architecture.md, threads | The timer thread also drains gsv's completion queue; compile workers are shared by translation and pipeline builds (M16) | graphics-server, shader-pipeline |
| `CLAUDE.md`, "Tests" | If render-backend decision 9 (reframed for wgpu) says yes: GPU tests required in Linux CI through wgpu on lavapipe, skipped locally with a notice when no adapter exists. The reverse of the hypervisor-test rule, so it needs its own line | render-backend decision 9 |

## 4. Work the drafts assume from elsewhere

No draft's port slices own these. Suggested owners in the last column.

| Work | Assumed by | Suggested owner |
|---|---|---|
| Kernel side of `WriteWatch`: applying protections requested from a non-core thread, the TLB flush, recording faults, restoring access | gpu-protocol (it says "a kernel slice, not listed here") | A new kernel slice, after slice 6 |
| Transfer memory (gsv 204, video 200) | graphics-server | Kernel "later slices" (shared and transfer memory); needs scheduling before graphics-server slice 5 |
| `svc 0x80` and the attribute range query; device-share lock and unlock | graphics-server slice 7 | Kernel, with slices 6 and 7 (the kernel spec lists `svc 0x80` under "Switch 2 specifics") |
| Event hook after lock release | graphics-server slice 5 (works without it, on the 1 ms bound) | Kernel slice 4 follow-up, or services |
| Lazy host storage for GPU-only memory | graphics-server says #12 and #14; gpu-protocol says gsv (M6) | **Owned since 2026-11-12:** gpu G22 |
| GPU side of ordered VM updates and barriers (`update_vm`, `barrier`, done reports) | graphics-server slices 7 and 16 (against a fake) | **Owned:** gpu G14 and G15 |
| Answering the shader pipeline: `EngineState`, `ProgramMemory`, building `VariantKey`, filling driver constants, calling `invalidate` | shader-pipeline ("What the GPU front/back protocol ... gets") | **Owned:** gpu G18 (widened), G16 for `invalidate` |
| Mapping render's `EmulationFlags` into the shader `VariantKey` | M13 | **Owned:** gpu G18 |
| The pipeline-key cache (guest state → `PipelineId`) and the list of keys to precompile next boot | render-backend ("Resources and who owns the caches") | **Owned:** gpu G23 |
| A wake-up for the back thread on host completion | gpu-protocol G15 (M14) | Still open; render-backend R5 once settled |
| The shared compile pool | M16 | **Owned:** render-backend R12 |
| Shader answers in the GPU capture | shader-pipeline | **Owned:** gpu G20 |
| The semaphore mirror page, once a trace settles row 35 | graphics-server, gpu-protocol | gpu G15 |

## 5. The decisions, in reading order

### A. Foundational: take these first

All six were decided on 2026-11-12 ("Decided so far"). The text below is kept for the record.

1. **Host API layer:** render-backend, decision 1 (decision 2, `ash`, follows it). It decides
   how many backends get written, what the Mac needs installed, whether swizzles become shader
   variants (M11), and the platform matrix. The draft recommends Vulkan with MoltenVK first and
   offers wgpu-first as a sound alternative.
2. **naga as the shader backend, and where final code is emitted:** shader-pipeline, decisions 2
   and 3, with seam M1. This fixes the shader → render handoff and which stages can reach any host
   (M12).
3. **When a GPU "done" becomes visible to the guest:** gpu-protocol, decision 2. It sets the meaning
   of every fence that gsv waits on, the present latch, and render's fence use. graphics-server
   assumed the other answer (M4).
4. **How presents pace the title:** graphics-server, decision 1. It shapes the present path through
   all three runtime drafts and decides M2 and M3.
5. **Where syncpoints and the translation table live:** graphics-server, decision 3, with M4 and M5.
   This is the main type seam between gsv and gpu; graphics-server slices 2 and 6 and gpu-protocol
   G7 build on it.
6. **Command lists as data, and guest caches in `gpu`:** render-backend, decisions 3 and 4. All
   drafts already assume both, so approve them together; every gpu-protocol cache slice rests on
   them.

### B. One question, asked in several drafts

Status after the six decisions: B1, B3 and B4 are open; B2 is implied by decision 4's wording
(the "emulated 60 Hz clock"), to be confirmed; B5 is settled in effect by decisions 1 and 2.
Render-backend decision 10 in B4 is implied by decision 2's wording ("the wgpu backend crate").

1. **A draw whose shader or pipeline is not ready:** gpu-protocol 9, shader-pipeline 6,
   render-backend 5, and the timing in shader-pipeline 7. All say wait by default (M10). Decide
   once, including when skip mode arrives.
2. **What paces the guest: emulated 60 Hz or the host display:** graphics-server 2 and
   render-backend 8. Both say the emulated clock from `Env`.
3. **When the references disagree:** graphics-server 8 (NX2 against PR #4588) and gpu-protocol 10
   (Mesa headers against Eden-NX2). Different pairs, compatible rules: each conflict is marked and
   pinned until a trace settles it. Accept both, or write one house rule.
4. **Which new crates enter the architecture:** gpu-protocol 5 (`starquake-layout`), render-backend
   10 (one crate per backend), shader-pipeline 14 (no separate decoder crate). Take with M8 and the
   `CLAUDE.md` changes in section 3.
5. **Geometry, tessellation, transform feedback:** shader-pipeline 13 and 3, and the stage rows of
   render-backend 1. All defer; M12 lists the two catches.

### C. Coupled: choosing one forces or favours another

Status after the six decisions: the first item (swizzles) is resolved by decision 1; render 9
is reframed for wgpu; render 7 is moot until a direct backend; render 12 holds under decision 2
(WGSL is wgpu's own language). The rest are open.

- **render-backend 1 ↔ shader-pipeline 16:** wgpu (option e) puts swizzles into the variant key,
  against shader-pipeline 16's "types and structure only" (M11).
- **shader-pipeline 5 ↔ gpu-protocol's translation interface:** option c requires gpu-protocol to
  take the non-blocking `lookup` (M9).
- **graphics-server 4 ↔ M6:** lazy host backing needs `gpu` to own the store.
- **graphics-server 12 ↔ gpu-protocol "Back-pressure":** graphics-server wants a bounded queue that
  defers the submit reply (`Submitted::Full`, `GpuQueueRoom`). gpu-protocol keeps submission queues
  unbounded, only flagging one longer than 65,536 entries. Choosing bounded means `Gpu::submit`
  can return "full". Recommended: bounded, at gpu-protocol's 65,536 figure, since host memory
  stays bounded and the deferred-reply path already exists in the service framework.
- **gpu-protocol 3 ↔ kernel `WriteWatch` work:** option b (trap CPU semaphore writes) needs the
  unowned kernel slice in section 4; option a (poll) needs nothing.
- **gpu-protocol 4 ↔ render-backend 7 ↔ shader-pipeline 9:** the unified-memory cluster. Aliasing
  guest memory as GPU buffers (render 7) makes GPU writes visible with no write-back (gpu 4). Real
  pointers for untraced global memory (shader 9, option b) need that aliasing and a non-naga
  writer. All three recommend the cheap option first; decide them together when revisited.
- **shader-pipeline 8 ↔ render-backend "Deferred":** true bindless (option b) needs descriptor
  indexing on Vulkan and argument buffers on Metal, which render-backend defers.
- **gpu-protocol 8 ↔ render-backend R11 ↔ shader-pipeline 7 and slice 20:** three capture formats
  (M15). Choose names and contents together.
- **graphics-server 9 ↔ kernel event hook:** on-demand answers need the unowned hook; until then
  the 1 ms bound alone.
- **shader-pipeline 11 ↔ render-backend 1:** mesh after milestone 3 either way, but wgpu's mesh
  support is experimental and the naga MSL fact is disputed (M12).
- **render-backend 9 ↔ `CLAUDE.md` "Tests":** lavapipe in CI needs the new skip rule (section 3).
- **render-backend 12 ↔ M1:** WGSL helper shaders keep naga in the render crate under either M1
  option.

### D. Local: take per draft, or accept as recommended in bulk

- **graphics-server** 5 (registration maps at its own address), 6 (one server per family), 7 (log
  the bootstrap order first), 10 (no video decoding at first, then FFmpeg off by default), 11 (only
  the nvdrv ioctl subset).
- **gpu-protocol** 1 (the front owns the register file; the largest internal choice), 6 (generate
  class tables from Mesa headers), 7 (Ampere classes only), 11 (one macro store per process), 12
  (drop same-value writes), 13 (stop on structural stream errors).
- **shader-pipeline** 1 (own arena IR with Eden's semantics; shapes the whole crate), 4 (strict
  decoding), 10 (ignore rounding modes except on conversions), 12 (shader overrides after milestone
  3), 15 (build the IR interpreter).
- **render-backend** 2 (`ash`: deferred by decision 1 until a direct Vulkan backend), 6 (ASTC to
  RGBA8 first), 11 (`--window` flag), 13 (GPUI frames decided later).

## 6. What can start before approval

Slices that depend on no open decision, no seam above and no other slice. All are pure and test
offline:

| Draft | Slice | Why it is free |
|---|---|---|
| gpu-protocol | G1: GP entry and method header decoding, push-buffer encoder | Bit fields from `clc76f.h`; no seam touches it |
| gpu-protocol | G2: acquire predicates, release sizes, reductions | Pure tables; the disputed QMD bit-19 rule is in G12, not here |
| gpu-protocol | G6: MME memo | Builds on the existing `gpu::mme` with the interpreter as oracle. gpu-protocol 11 (macro store per process) changes only when the memo is cleared |

Free after one local decision, with no seam involved:

- gpu-protocol **G3** (class-table generator), once gpu-protocol 6 and 7 are taken.
- gpu-protocol **G5** (`starquake-layout`), once gpu-protocol 5 is taken. Its first test also
  closes render-backend verification 5.

Not free: every graphics-server slice (slices 2 and 6 no longer wait on a decision or seam,
since decision 3 and M4–M5 are settled, but still on the draft's approval; the rest depend on
slices 1, 2 or 3); every shader-pipeline slice (all depend on slice 1, which waits on decision 1
and experiment E1); every render-backend slice (R0's seams M1, M3, M13 and M14 are now settled in
the drafts, except the completion wake, but the draft is not approved, and every other slice
depends on R0). Experiments need no approval and can run now: shader-pipeline E1–E3 (E1 now
includes creating the modules on wgpu on a Mac), and render-backend verifications 1, 2 and 3
(verification 1 is a few lines printing wgpu's adapter features and limits; nothing to
install).
