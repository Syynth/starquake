# The hello fixture as a reference emulator runs it

`startup.trace` is what `../hello.nro` does at the system boundary from its first instruction
to `ExitProcess`, recorded from a reference emulator and normalised so that another emulator's
run can be compared with it. `crates/core/tests/homebrew.rs` runs the fixture in starquake and
compares its trace with this one, call by call, up to `ConnectToNamedPort("sm:")`; the rest of
the file is what the next slices have to match.

## Provenance

| | |
|---|---|
| Emulator | Eden-NX2 (`Takoroka/eden-nx2`, branch `eden-nx2`) at commit `50fa86bbbc40`: the source `docs/references.md` pins as `eden-nx2` |
| Build | Its `eden-cli`, built on macOS (Apple Silicon) from that commit with one local patch, which moves a window-title update to the main thread (`src/yuzu_cmd/emu_window/emu_window_sdl3.cpp`) and touches nothing the guest can see |
| Run | `eden-cli -n -s -g hello.nro` (null renderer, single core), with the emulator's own trace switches `EDEN_NX2_SVC_TRACE=1` and `EDEN_NX2_IPC_TRACE=1`, default settings otherwise (a 4 GiB console) |
| Program | `../hello.nro`, SHA-256 `d21c81c6e795ab9981e0757bf2b4b23184e974c30ded266e3098a468e7fb20df` (the hash `../manifest.toml` records) |
| Recorded | 2026-11-11, three runs |

The raw traces are not in the repository: they hold the emulator's randomised addresses and
its timestamps, and nothing else that is not here. The three runs differ in exactly those and
normalise to the same text, byte for byte, which is the check that the normalisation masks
everything that varies and nothing else.

## Format

`normalise.py` (in this folder) wrote the file; its header says how. One line per event, in the
order the emulator logged them, all on the program's one thread:

```text
svc 0x02 SetMemoryPermission(address=main+0x4000, size=0x1000, perm=0x1) -> 0x0
svc 0x29 GetInfo(info_type=0x2, handle=0xffff8001, info_subtype=0x0) -> 0x0 out=alias_region+0x0
ipc sm:::0 (Initialize)
log starquake hello: line one
```

- `svc <number> <name>(<arguments>) -> <result> <outputs>`: a system call. Arguments and
  outputs are named and decoded with the kernel's register convention
  (`crates/kernel/src/svc/reference/svc_wrappers64.txt`), so registers a call does not take
  are not there. Outputs other than the result follow it, and are left out when the call
  failed. A call with no result (`ExitProcess`) has no `->`.
- `ipc <service>::<command> (<name>)`: an IPC request, as the emulator names it, logged while
  the `SendSyncRequest` that carries it is in progress, so before that call's line.
- `log <text>`: a line the guest wrote with `OutputDebugString`, before that call's line.
- An address is `anchor+offset`. `main` is the module's first byte. `aslr_region`,
  `stack_region`, `alias_region` and `heap_region` are the starts of the regions `GetInfo`
  reported in the same run. An emulator that places a region elsewhere gets the same text;
  one that places something differently *inside* a region (the stack four pages into the
  stack region, say) does not.
- Sizes, handles and results are as the emulator returned them. Handle values
  (`out_handle=0x103fe`) are that emulator's; nothing compares them yet.

## Regenerating

With a raw `startup.txt` from a run of the same emulator (its `svc`, `ipc` and guest-log lines):

```sh
tests/homebrew/hello/reference/normalise.py <raw startup.txt> > normalised.txt
```

and put the file's five header lines back in front. The script refuses a trace it does not
fully understand (a line it cannot parse, a call it has no address rule for, a second thread,
an address in no known region), so a new fixture needs its rules extended, not loosened.
