# oboromi as a base for the fork: evaluation

Written 2026-11-11 from `0xNikilite/oboromi` @ `c97664a` (2026-11-03, 304 commits) and
`vrtgs/rapid-arm-emu` @ `d762465` (2026-08-02, 36 commits). File references are to those trees.
Read alongside `switch2-eden-diff-scoping.md` and `docs/decision-log.md`.

## Verdict

**Don't fork oboromi as the base. Start a fresh GPLv3 workspace and borrow two crates from
rapid-arm-emu (the A64 decoder and, later, the software MMU).** Nothing in oboromi itself is worth
keeping. Its name and its 1.4k stars are its main assets, not its code.

- oboromi is a **scaffold, not an emulator**. It has about 7k lines of hand-written Rust, most of
  them `todo!()` stubs. It has no loader, no kernel or SVCs, no IPC dispatch, no GPU command
  processor and no rendering. Its CPU is a thin wrapper around Unicorn (QEMU TCG), which the
  maintainer plans to replace.
- None of oboromi's structure fits the decisions already made: pluggable CPU backends, Vulkan plus
  Metal, and agent-driven development. Every piece would be rewritten, so a fork only buys history
  and a C/C++ build (Unicorn, Qt, CMake) that we'd delete first.
- **rapid-arm-emu is the better-engineered code.** It has strict lints, Miri-checked memory code
  and a full A64 decoder generated from Arm's 2026-03 XML. It doesn't execute any instructions yet,
  but its decoder is directly useful for the load-time code patcher that HVF and KVM need.
- This revisits the "fork, most likely of oboromi" decision. That entry was explicitly "pending
  the investigation". A fresh GPLv3 repo keeps everything that decision wanted: ownership, our own
  direction, and the freedom to port from Eden and oboromi.

## 1. Architecture: what exists and what runs

One crate, `oboromi-core` ([core/Cargo.toml](core/Cargo.toml)). It builds as an rlib plus a staticlib
for a Qt 6 C++ frontend ([gui-qt/](gui-qt/), [CMakeLists.txt](CMakeLists.txt), using Corrosion).

| Module | What's there | How far it goes |
|---|---|---|
| `cpu/` | `UnicornCPU` wrapper ([unicorn_interface.rs](core/src/cpu/unicorn_interface.rs)) and `CpuManager` with **8 cores** over a flat 12 GiB `Vec<u8>` mapped at VA 0 ([cpu_manager.rs:4-50](core/src/cpu/cpu_manager.rs)) | Runs hand-assembled ADD/SUB/MOV/RET snippets until a `BRK`. It has no exception, SVC or abort handling, no system registers, no page tables and no permissions. `run_all` single-steps each core in turn. |
| memory | none beyond the flat buffer | No address-space model, so a 39-bit sparse layout or 20 GiB+ reservations aren't possible. |
| loader / `fs/` | `fs::File` is an mmap wrapper ([fs/mod.rs](core/src/fs/mod.rs)) | No NSO, NRO, NPDM, NCA or RomFS. |
| kernel / HLE | none | No SVC table, threads, handles or sync. |
| `nn/` services | 160 service names expanded by a macro into **empty `State {}` structs** ([nn/mod.rs:6-51](core/src/nn/mod.rs)), held as `Option`s in `sys::Services` ([sys/mod.rs](core/src/sys/mod.rs)) | The names are a useful checklist (`gsv`, `syncpt`, `vi2` and `nvdrv` are on it). There are no commands. |
| IPC | `hipc.rs` decodes the 8-byte HIPC header and ignores every field ([nn/hipc.rs:41-62](core/src/nn/hipc.rs)). `build.rs` would generate method stubs from `nn/auto.defs`, but that file is empty. | No CMIF or TIPC framing, no descriptors, no sessions. |
| `gpu/` | `VkState` creates an ash device and a compute queue ([gpu/mod.rs](core/src/gpu/mod.rs)). There's a hand-written SPIR-V emitter ([spirv.rs](core/src/gpu/spirv.rs), 1.1k lines) and the SM86 decoder (§3). | No GPFIFO, engines, MME, memory manager, textures or presentation. |
| `tests/` | GUI-driven suites that return `"Y …"`/`"N …"` strings, re-asserted by `cargo test` ([tests/suites.rs](core/src/tests/suites.rs)) | Smoke level only (§5). |

**What runs today:** single-instruction smoke tests on Unicorn, and SM86-to-SPIR-V translation of
roughly six instruction forms into a module that is only structurally checked. Nothing loads a
binary. The README says as much ("cannot currently run commercial games or firmware").

## 2. CPU

**Today it uses Unicorn 2.1 (QEMU TCG), through `unicorn-engine = "2.1.1"`.** That's a JIT, not
an interpreter. The CPU has been rewritten twice: dynarmic from 2025-08 to 2025-11 (the
`.gitmodules` entries for `third_party/dynarmic` and `ext-boost` are stale leftovers), then
Unicorn. The maintainer turns away dynarmic PRs (#88) and alternative CPUs (#90). In PR #58 they
said "we're moving to LLVM for the CPU backend", meaning rapid-arm-emu.

**ARMv8.x/v9 coverage in oboromi:** none of its own. The code never chooses a Unicorn CPU model,
so it gets Unicorn's AArch64 default (Cortex-A72-class, ARMv8.0, as I recall Unicorn 2.x
behaving). That means none of LSE, RCpc, PAuth or MTE, all of which Switch 2 games use heavily
(scoping doc, "CPU and ISA"). Nothing tests v8.1+ instructions.

**rapid-arm-emu** (separate repo, GPLv3, a single author, last commit 2026-08-02):

| Crate | State |
|---|---|
| `arm-instruction-decoder` | Generated from Arm's machine-readable A64/A32/T32 XML (`spec/`, 2026-03 release). It covers the whole ISA, including the Switch 2 ones: `LDAPR`, `RETAA`/`RETAB`, `CAS`/`CASP`, `SWP*`, `LD<op>` LSE, `PACGA`/`XPAC*`, MTE `STG`, BTI and SVE. It's `no_std` with no dependencies. Its test passes (5.4 s). |
| `io-mmu` | A software page-table MMU over the full 64-bit space, with a TLB/lookup cache, an exclusive monitor and demand-paged memory objects. **75 integration tests pass**, and it's Miri-tested. |
| `jit-compiler-core` | An SSA IR with halt-check/safepoint passes, and Cranelift and GCC-JIT backends (LLVM is commented out). It **links `libgccjit` unconditionally**, so its tests didn't link on this Mac. |
| `rapid-arm-emu` | `Armv9CpuCore::resume() -> HaltReason` has the right shape, but `CodeCache::run` is `todo!()` ([armv9/jit.rs](rapid-arm-emu/src/armv9/jit.rs)) and `front_end/` is empty. **No instruction is lifted to IR yet**, and it isn't wired into oboromi. |

**Adding a backend trait.** In oboromi there's nothing to put the trait behind. `UnicornCPU` is a
concrete type that `CpuManager` stores directly. Its API is register access, `step` and run until
`BRK`. Memory is accessed through the CPU object and lives at VA 0 with no page tables. The trait
we need is shaped around exits:

```rust
trait VcpuBackend {
    fn run(&mut self) -> Exit; // Svc(imm) | DataAbort{..} | Undefined | Timer | Halt
    fn regs(&mut self) -> &mut GuestRegs;
    fn sysreg(&mut self, r: SysReg) -> u64; // TPIDR, CNT*, …
}
```

It also needs a guest-memory abstraction that HVF and KVM implement with stage-2 mappings and a
JIT implements with a software MMU. That is a design from scratch whether or not we fork; about a
day to write, and none of oboromi's code survives it. rapid-arm-emu's `resume() -> HaltReason`
plus `IoMMU` would be the x86-64 JIT side of that trait once it actually executes code. On
HVF/KVM `io-mmu` doesn't apply, since the hardware translates; its decoder is still what we'd use
to patch `mrs cntpct_el0` and `svc` (and, if PAuth isn't available, `RETAA`) at load time.

## 3. GPU and shaders

- **SM86 decoder** ([sm86.rs](core/src/gpu/sm86.rs), 4.2k lines; [sm86_decoder_generated.rs](core/src/gpu/sm86_decoder_generated.rs),
  2.5k lines). `translate()` is a generated **chain of 1,271 `if/else` arms**, including duplicate
  arms, dispatching to 201 handlers. Only `al2p`, `exit`, `iadd32i`, `kill` and `nop` are complete.
  `iadd3` and `mov` handle the register and immediate forms and `todo!()` on constant-bank and
  uniform operands. Every other handler (FFMA, LDG, TEX, BRA, …) ends in `todo!()`, and unknown
  encodings panic (#69). There is no control-flow recovery, no IR and no optimisation. The decoder
  writes SPIR-V straight into a private register array.
- **Translation target:** hand-rolled SPIR-V. `validate()` only checks word counts, and CI doesn't
  run `spirv-val`.
- **Rendering API:** Vulkan 1.2 through `ash`, used only to create a device. There is no swapchain,
  pipeline or present path. **There's no Metal path.** On macOS it would mean MoltenVK, which
  works against the scoping doc's unified-memory, ASTC and sparse-heap advantages.
- **Both Vulkan and Metal:** nothing here helps. The scoping doc's plan, an SM86 IR lowered to a
  `naga::Module` that emits SPIR-V and MSL, has to be built either way.
- **Provenance risk:** the decoder is generated from
  [scripts/sm_86_instructions.txt](scripts/sm_86_instructions.txt) (120k lines). It looks like
  NVIDIA's internal machine description: it starts `ARCHITECTURE "Volta"` and has `R_CUDA_*`
  relocator tables and pipe classes. A PDF rendering of it sits at [docs/sm86.pdf](docs/sm86.pdf),
  and there's a latency table too. All of it is redistributed in the repo. Mesa NAK's MIT-licensed
  `sm70_encode.rs`, checked against `nvdisasm` output, is a cleaner source. This isn't legal
  advice.

## 4. Portability today

| | oboromi | rapid-arm-emu |
|---|---|---|
| Linux x86-64 / arm64 | CI builds on `ubuntu-latest`. Unicorn and ash are portable, and the GPU tests use lavapipe. | Cranelift handles x86-64 and arm64. The build needs `libgccjit`. |
| macOS arm64 | **The build fails on this Mac.** `unicorn-engine-sys 2.1.5`'s bundled QEMU won't compile against the current SDK (`Int128` typedef clash, missing `PROT_*`). CI passes on GitHub's `macos-14` and `macos-latest`. The CPU suite is `#[ignore]`d on macOS. | Builds. JIT tests fail to link without `brew install libgccjit`. |
| Windows | In CI. A 12 GiB `vec![0; …]` is committed up front ([cpu_manager.rs:26-28](core/src/cpu/cpu_manager.rs)), and the maintainer says Unicorn's Windows build is broken (#58). | Not tried. |
| Native execution (HVF/KVM) | None | None |

## 5. Fitness for agent-driven work

- **Tests:** about 25 in total: HIPC header (8), `fs` (3), SPIR-V literals/validate, Vulkan init
  or skip, multicore memory sharing (2), and the GUI suites (9 CPU snippets, 6 SM86 forms). Nothing
  is differential (against hardware, nvdisasm or a reference CPU), nothing uses recorded traces and
  there are no fixtures.
- **CI:** five workflows with about 900 lines of YAML (build matrix, GUI builds on Qt 6.12,
  release, cargo-audit, a "PR safety check"). **There's no `clippy` or `rustfmt` gate.**
  CONTRIBUTING says fmt is skipped "for now". Recent PR runs fail about half the time; the last
  `Rust CI` run (2026-11-03) failed.
- **Build speed:** dominated by the Unicorn/QEMU C build (cmake) and, for the GUI, Qt plus
  Corrosion. rapid-arm-emu's decoder plus MMU tests built in 21 s from cold.
- **Code quality and consistency:** low and uneven.
  - Comments left over from earlier CPUs ("Dynarmic JIT backend", "JIT warmup" in a Unicorn test
    runner).
  - `unsafe impl Send/Sync` on the Unicorn wrapper.
  - Raw pointers in `gpu::State`.
  - Empty service structs.
  - Duplicate decoder arms.
  - Tests that pass by string-matching `"Y "`.

  rapid-arm-emu is the opposite: `arithmetic_side_effects`, `cast_*`, `missing_docs` and
  `missing_safety_doc` are denied, `unsafe_op_in_unsafe_fn` is forbidden, and the MMU is Miri-run.
  Even so, it is WIP: its commits include "terrible code quality" and "WIP" messages, and it has
  no CI.
- **Docs:** README, CONTRIBUTING and the SM86 PDF. There are no design docs, specs or protocol
  notes.
- **Changing it with confidence:** low. Few invariants are encoded, little is tested, and the
  important seams (backend, memory, IPC) don't exist yet.

## 6. Project health

- **Activity:** bursty. There were 36–53 commits a month in mid-2025, then a near-stop
  (2025-12 to 2026-09 averages about 5 a month), and a revival in 2026-10 (31). The last commit
  was 2026-11-03.
- **Contributors:** one maintainer (Nikilite, about 230 commits under three identities), then
  xinitrcn1 (11), lizzie of Eden (9), jopdorp (13) and a long tail. There are 1,405 stars and 68
  forks.
- **Review culture:** strict scope control. jopdorp's stacked series (#78–#88, about 44k lines
  including a Vulkan pipeline, a GPU command processor and audio) and several large outside PRs
  (#66, #90) were closed as unreviewable or off-direction. **Those closed PRs are the closest thing
  to a working GPU path in oboromi's orbit.** They're worth reading, but none of them is in the
  tree.
- **Licence:**
  - [license.txt](license.txt) is the plain GPLv3 text. `Cargo.toml` says `GPL-3.0` (the deprecated
    SPDX id, which means GPL-3.0-only).
  - **There's no "or any later version" grant** anywhere: no file headers, no SPDX tags.
  - The project was MPL-2.0 until 2025-11-19 (`67864b0`). Earlier outside contributions came in
    under MPL-2.0. MPL-2.0 is GPL-compatible, so that only matters if someone wanted to relicence.
  - There's no CLA. CONTRIBUTING states inbound = outbound GPLv3.
  - rapid-arm-emu has a GPLv3 text file, and its `Cargo.toml` uses the invalid id `GPL-3`.
  - In practice, treat both as **GPL-3.0-only** unless the authors add an "or later" grant.
- **History:** linear and readable. Commits are small and tagged (`[gpu]`, `[ci]`), apart from
  the 2025 dynarmic period. The repo carries large data files (the SM86 PDF and the 120k-line
  table), and rapid-arm-emu carries Arm's full PDFs and XML tarballs (LFS for the generator
  inputs).

## 7. Recommendation

**Borrow pieces into a fresh GPLv3 repo.** Don't fork oboromi as-is, and don't fork and
restructure it.

Why:
1. **There's nothing to restructure.** A fork that restructures deletes `cpu/`, `nn/`, `sys/`,
   `gui-qt/`, CMake and Unicorn, and rewrites `gpu/`. What's left is a name and history whose
   build we'd have just broken on purpose.
2. **The C/C++ build works against the fast gate in decision 3.** It also fails on this machine
   today.
3. **The SM86 tables are a provenance liability** we'd inherit by forking. A fresh repo can build
   its decoder from NAK (MIT) and nvdisasm-checked fixtures.
4. **GPLv3 is already the plan**, so we can still copy anything useful from oboromi or Eden later,
   with attribution.

Worth borrowing:
- **`arm-instruction-decoder` from rapid-arm-emu.** It's the decoder for the load-time patcher
  (`cntpct`, `svc`, PAuth fallback) and, later, the x86-64 JIT frontend. Vendor it by commit with
  its licence and attribution.
- **`io-mmu`, later,** as the guest-memory implementation behind the JIT backend.
- **oboromi's service-name list** as a checklist, and **jopdorp's closed PRs** for ideas.

If the user still prefers a literal fork for continuity, fork **rapid-arm-emu** (clean, strict,
CPU-shaped) rather than oboromi. That's a separate decision, though.

First changes, in order:
1. **Skeleton and gate.**
   - Create a cargo workspace using the scoping doc's crate layout (`hos-loader`, `hos-cpu`,
     `hos-kernel`, `hos-ipc`, …).
   - Add one `cargo xtask ci` that runs `fmt --check`, `clippy -D warnings` (with rapid-arm-emu's
     lint set) and `nextest` in a few minutes on macOS arm64 and Linux x86-64.
   - Keep C/C++ dependencies out of the default build.
2. **`hos-cpu` backend trait plus two backends.** HVF first, building on the main session's
   prototype. Then a deterministic test backend: either rapid-arm-emu once it executes, or
   Unicorn behind a feature flag. Define `Exit`, register and sysreg access, and a `GuestMemory`
   trait (stage-2 for HVF/KVM, software MMU for a JIT). Leave the KVM and x86-64 JIT slots stubbed
   but compiling.
3. **Loader and a libnx hello-world harness** (milestone 1). Cover NSO/NRO/NPDM with both ACID
   layouts, plus a load-time patcher built on the vendored A64 decoder. Golden tests record the
   SVC and IPC trace of a homebrew run, so agents get a deterministic, keyless regression test.
4. **`hos-ipc` CMIF and TIPC framing with property tests**, covering the TIPC `Push(Result)` bug
   from the scoping doc. Add auto-stub plus tracing for unknown commands.
5. **The SM86 shader-test harness, before the translator.** Use encodings from NAK, per-instruction
   fixtures checked against `nvdisasm`, and IR to `naga` to `spirv-val` and MSL compile checks in
   CI. Then grow instruction coverage against that harness.
