# The hello fixture as a reference emulator runs it

`startup.trace` is what `../hello.nro` does at the system boundary from its first instruction
to `ExitProcess`, recorded from a reference emulator and normalised so that another emulator's
run can be compared with it. `crates/core/tests/homebrew.rs` runs the fixture in starquake and
compares its trace with this one, line by line, through `ExitProcess`; its module docs say how
each kind of line is compared.

## Provenance

| | |
|---|---|
| Emulator | Eden-NX2 at commit `50fa86bbbc40` (`eden-nx2` in `docs/references.md`), built and run as `docs/reference-emulator.md` says, with the one patch it names (`tools/oracle/eden-nx2-oracle.patch`: a window-title fix, nothing the guest can see) |
| Run | `tools/oracle/run-nro.sh` (null renderer, one core, the emulator's own system-call and IPC trace switches, default settings otherwise: a 4 GiB console), then `tools/oracle/summarize.py`, whose `startup.txt` is what `tools/oracle/normalise.py` reads |
| Program | `../hello.nro`, SHA-256 `d21c81c6e795ab9981e0757bf2b4b23184e974c30ded266e3098a468e7fb20df` (the hash `../manifest.toml` records) |
| Recorded | 2026-11-11, three runs |

The raw run folders are not in the repository (`docs/reference-emulator.md`, "What may be
committed"): they hold the emulator's randomised addresses and its timestamps, and nothing
else that is not here. The three runs differ in exactly those and normalise to the same text,
byte for byte, which is the check that the normalisation masks everything that varies and
nothing else.

What the reference is trusted for, and where it is known to be wrong, is in
`docs/reference-emulator.md`. Two of its notes concern this file: the two `SendSyncRequest`
lines with result `0xf601` are its answer to a session's close request, which is unverified;
and its `ExitProcess` returned to the guest, which this format does not show.

## Format

`tools/oracle/normalise.py` wrote the file; its module docs say how. One line per event, in the
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
  (`out_handle=0x103fe`) are that emulator's; the comparison pairs each with the handle
  starquake used in the same place and checks that the pairing holds for the whole run.

## Regenerating

With the `startup.txt` of a run (`docs/reference-emulator.md`, "Running it"):

```sh
tools/oracle/normalise.py --nro tests/homebrew/hello/hello.nro <run-folder>/startup.txt > normalised.txt
```

and put the file's five header lines back in front. The script refuses a trace it does not
fully understand (a line it cannot parse, a call it has no address rule for, a second thread,
an address in no known region), so a new fixture needs its rules extended, not loosened.
