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

All the way: `starquake run tests/homebrew/hello/hello.nro` exits with code 0, as of the kernel's
sessions (#98, milestone 1). Every system call, IPC request and log line matches the reference
emulator's, in order, through `ExitProcess` (`hello/reference/`, and
`crates/core/tests/homebrew.rs`):

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
  svc 0x1f(0x0, 0x80003148) -> 0x0
  ipc session=4294967297 service=1 command=3 req=8 rep=2 -> 0x0 Done
  svc 0x21(0x103fe) -> 0x0
  ipc session=4294967297 service=1 command=0 req=12 rep=0 -> 0x0 Done
  svc 0x21(0x103fe) -> 0x0
  log len=25 hash=0xb51b0fbfdb3b5bd2
  svc 0x27(0x800030a0, 0x19) -> 0x0
  log len=25 hash=0x8dab36c055e5ad88
  svc 0x27(0x800030c0, 0x19) -> 0x0
  ipc session=4294967297 service=1 command=1 req=16 rep=0 -> 0x0 Done
  svc 0x21(0x103fe) -> 0x0
  ipc session=4294967298 service=2 command=3 req=8 rep=2 -> 0x0 Done
  svc 0x21(0x183fd) -> 0x0
  ipc session=4294967298 service=2 command=3 req=8 rep=0 -> 0x0 Done
  svc 0x21(0x183fd) -> 0x0
  svc 0x21(0x183fd) -> 0xf601
  svc 0x16(0x183fd) -> 0x0
  log len=44 hash=0x7c7d5ae1b34ce652
  svc 0x27(0x800030e0, 0x2c) -> 0x0
  svc 0x21(0x103fe) -> 0xf601
  svc 0x16(0x103fe) -> 0x0
  svc 0x7() -> 0x0
  lifecycle ProcessExit 1
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
4. The fixture's `__appInit` calls `smInitialize`: `svcConnectToNamedPort` (0x1f) on `"sm:"`,
   then `svcSendSyncRequest` (0x21) twice, for the pointer buffer size (a control request,
   which starquake's trace shows as command 3 on `sm:`, service 1) and for `Initialize`.
5. `main` writes its two lines with `svcOutputDebugString` (0x27), asks `sm:` for `set:sys`
   (`GetService`, command 1, which moves a session handle back), asks `set:sys` for its pointer
   buffer size and the firmware version (`GetFirmwareVersion`, command 3, which writes into a
   receive buffer), and closes it: a close request, which ends with the session closed
   (0xf601), then `svcCloseHandle` (0x16).
6. It writes "firmware version ok", returns from `main`, and libnx closes `sm:` the same way and
   calls `svcExitProcess` (0x07), which ends the run.

`crates/core/tests/homebrew.rs` runs the fixture and compares its trace with the reference's
only in its `on_hardware` module, which is compiled for macOS on Apple Silicon and returns early
when the `CI` environment variable is set: so on a gate run on a Mac. On the hosted CI gate only
its `comparison` module runs, which checks the comparison itself against made-up runs; the same
conversation with `sm:` and `set:sys` runs on every host without a vCPU in
`crates/core/src/hle/tests.rs`.

## `startup/`

A libnx program built the ordinary way (`startup.c`): it keeps libnx's own `__appInit`, so
start-up opens `sm:`, then `set:sys`, `apm`, `appletOE`, `hid`, `time:u`, `fsp-srv` and `vi:m`
before `main` (`libnx/nx/source/runtime/init.c` and `display/default_window.c` @ b156b477).
`main` prints one line on the console, which opens `nvdrv` for the framebuffer, and returns;
libnx then closes everything and ends the process through `appletOE`.

| File | What it is | Built by |
|---|---|---|
| `startup.c` | The program | |
| `Makefile` | Builds the NRO and writes `manifest.toml`, with the hello fixture's flags | |
| `startup.nro` | What `starquake run` loads, with an `ASET` block holding a NACP | `elf2nro` |
| `manifest.toml` | Toolchain versions, the ELF's build id, and the SHA-256 of every source and binary | `make` |
| `reference/` | The reference emulator's normalised trace and its README, with the gap list | `tools/oracle/` |

`crates/loader/tests/homebrew_fixture.rs` checks the manifest as it does the hello fixture's.
Rebuild with `make -C tests/homebrew/startup` (same set-up as above).

### How far `startup.nro` runs

Not far yet, and that is the point of it: starquake's run matches the start of the reference
trace (`startup/reference/startup.trace`), and `crates/core/tests/homebrew.rs` fails if a change
makes it match fewer lines than its `STARTUP_MATCHED` records (the change that makes it match
more raises that constant). When the fixture was added (#113), the run matched start-up,
`set:sys`, and `sm:`'s `GetService` for `apm`, and differed at the next request: starquake has
no `apm`, so `sm:`'s auto-stub hands out a session whose requests are logged under `sm:`'s
number, where the reference has `apm`'s. `startup/reference/README.md` lists every system call
and request in the trace that starquake does not implement yet, in order of first use, written
from the trace by the same test file.
