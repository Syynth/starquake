# Homebrew fixtures

Real programs built with devkitPro's Switch toolchain, committed as binaries next to their source
(test strategy spec, `docs/specs/testing.md`, "Homebrew fixtures", decision 1). The gate never
builds them, so it doesn't need the toolchain.

## `hello/`

The smallest useful libnx program (`hello.c`): it writes two known lines with
`svcOutputDebugString`, asks `set:sys` for the firmware version, writes a line saying whether that
worked, and exits. It overrides libnx's `__appInit` so start-up opens only `sm:`
(`docs/specs/services.md`, "Milestone 1 services").

| File | What it is | Built by |
|---|---|---|
| `hello.c` | The program | |
| `hello.json` | Its NPDM description: title id, main thread, service list (one exact name and one `*` wildcard), kernel capabilities | |
| `Makefile` | Builds the three binaries and writes `manifest.toml` | |
| `hello.nro` | What `starquake run` loads, with an `ASET` block holding a NACP | `elf2nro` |
| `hello.nso` | The same ELF as an NSO, for the loader tests | `elf2nso` |
| `main.npdm` | The NPDM | `npdmtool` |
| `manifest.toml` | Toolchain versions, and the SHA-256 of every source and binary | `make` |

`crates/loader/tests/homebrew_fixture.rs` checks the manifest on every gate run: a source or a
binary whose hash differs, a file in the folder that the manifest does not list, or a missing
toolchain version fails the gate. That catches a source edited without a rebuild (or a binary
replaced without its source). It does not prove that the binaries were built from these sources;
rebuilding and comparing is the manual check (the build is reproducible: rebuilding with the
versions in the manifest gives the same bytes).

The same test file reads the binaries with the loader and checks what the toolchain wrote against
`hello.json` and against each other (the NRO and the NSO come from one ELF).

## Rebuilding

Install the toolchain once with `tools/setup-devkitpro.sh` (macOS; it asks for your password),
then:

```sh
export DEVKITPRO=/opt/devkitpro
make -C tests/homebrew/hello
```

Commit the changed sources, binaries and `manifest.toml` together. `npdmtool` prints
"Failed to get ... (field not present)" for every optional field `hello.json` leaves out; that is
expected.

## How far `hello.nro` runs

`starquake run tests/homebrew/hello/hello.nro`, as of the kernel's address space (#97). The run
now does everything the reference emulator's does before IPC, call for call
(`hello/reference/`, and `crates/core/tests/homebrew.rs`, which compares the two on every gate
run), and stops at the first call that needs a session:

```text
thread 1
  lifecycle ProcessStart 1
  svc 0x2(0x80004000, 0x1000, 0x1) -> 0x0
  svc 0x3(0x80004000, 0x1000, 0x10, 0x10) -> 0x0
  svc 0x29(0x0, 0x2, 0xffff8001) -> 0x0
  ... ten more svc 0x29, for ids 0x3, 0x1c, 0x4, 0x5, 0xc, 0xd, 0xe, 0xf, 0x6, 0x7
  svc 0x1(0x0, 0xcd000000) -> 0x0
  svc 0x6(0x1080303fc8, 0x0, 0x1080303fc8) -> 0x0
  svc 0x6(0x1080303fc8, 0x0, 0x80006000) -> 0x0
  svc 0x1f(0x0, 0x80003148) -> 0x4201
  svc 0x26(0x0, 0x1080303fdc, 0x4) -> 0x4201
```

1. libnx's start-up applies the module's relocations, then makes the RELRO range read-only with
   `svcSetMemoryPermission` (0x02) on `.data`'s first page and locks that permission with
   `svcSetMemoryAttribute` (0x03) (`libnx/nx/source/runtime/dynamic.c` @ b156b477, after the
   `LNY0`/`LNY1` check).
2. `virtmemSetup` asks `svcGetInfo` (0x29) for the alias, heap, ASLR and stack regions and the
   alias region's extra size (`libnx/nx/source/kernel/virtmem.c`), and `__libnx_initheap` for
   the total and used memory, from which it sizes its heap: `svcSetHeapSize` (0x01) with
   0xcd000000 (`libnx/nx/source/runtime/init.c`).
3. `svcQueryMemory` (0x06) twice: once about an address on its own stack, for the main
   thread's stack bounds (`libnx/nx/source/kernel/thread.c`), and once about the first address
   after its image, for arguments (`libnx/nx/source/runtime/argv.c`).
4. The fixture's `__appInit` calls `smInitialize`, whose `svcConnectToNamedPort` (0x1f) on
   `"sm:"` is not implemented and returns 0x4201. The fixture treats that as fatal:
   `diagAbortWithResult` calls `svcBreak` (0x26), which is not implemented either.
5. `diagAbortWithResult` does not expect `svcBreak` to return (`__builtin_unreachable()` in
   `libnx/nx/source/runtime/diag.c`), and the compiler put `udf #0` after the call. The run
   ends as a guest crash (exit code 1, undefined instruction) at `main+0xb5c`.

So the next missing piece is the session side of the kernel: `svcConnectToNamedPort` and
`svcSendSyncRequest`, and behind them `sm:` and `set:sys`. The guest log is still empty: the
fixture's first line comes after `smInitialize`.
