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

Recorded with `starquake run tests/homebrew/hello/hello.nro` on the walking skeleton (#74), as of
the commit that added this fixture. It stops at the first system call the skeleton lacks; nothing
was added to the kernel or services to get further.

```text
thread 1
  lifecycle ProcessStart 1
  svc 0x2(0x80004000, 0x1000, 0x1) -> 0x4201
  svc 0x26(0x0, 0x80ffffdc, 0x4) -> 0x4201
```

1. libnx's start-up applies the module's relocations, then makes the RELRO range read-only with
   `svcSetMemoryPermission` (0x02) on `.data`'s first page (`libnx/nx/source/runtime/dynamic.c`
   @ b156b477, after the `LNY0`/`LNY1` check). The skeleton has no handler for it and returns
   0x4201.
2. libnx treats the failure as fatal: `diagAbortWithResult` calls `svcBreak` (0x26), which is
   not implemented either and also returns 0x4201.
3. `diagAbortWithResult` does not expect `svcBreak` to return; the next instruction is undefined,
   and the run ends as a guest crash (exit code 1) at `main+0xb5c`.

So the first missing piece is `svcSetMemoryPermission`; the next one start-up reaches is
`svcSetMemoryAttribute` (0x03), which libnx calls right after it and whose result it ignores.
Nothing reached `sm:` or `set:sys`, and the guest log is empty.
