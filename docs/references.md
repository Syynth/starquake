# Reference sources

Starquake is ported from, and checked against, other projects' source. Those sources are **not**
in this repository. They live in one shared folder next to the main checkout, so every worktree
and agent reads the same copy:

```sh
tools/fetch-refs.sh     # or: just refs
tools/refs-path.sh      # prints the folder; STARQUAKE_REFS overrides it
```

| Folder | Project | Commit | Licence | Used for |
|---|---|---|---|---|
| `eden` | Eden mainline (`eden-emu/eden`, master) | `cf654052c416` | GPL-3.0-or-later | Kernel, IPC, services, loader, GPU command processing, shader IR |
| `eden-nx2` | Eden-NX2 (`Takoroka/eden-nx2`, branch `eden-nx2`) | `50fa86bbbc40` | GPL-3.0-or-later | Switch 2 behaviour: `NX2_NOTES.md`, gsv/syncpt/vi2, Turing macros, GPU fixes |
| `eden-pr4588` | Eden PR #4588 (`ponting/eden`, branch `ns2-support`) | `e5fb3b97c94b` | GPL-3.0-or-later | Native Ampere shader frontend and its tests, six cores, graphics supervisor protocol |
| `mesa-nouveau` | Mesa (`src/nouveau/{compiler,headers,nil,mme}` only) | `52238762dd44` | MIT | NAK shader encodings, NVIDIA class headers, image layout, MME simulator |
| `libnx` | libnx (`switchbrew/libnx`, branch `master`) | `b156b4774fb2` | ISC | The client side of IPC: how a homebrew program lays out CMIF, domain and TIPC requests (`nx/include/switch/sf/{hipc,cmif,tipc}.h`) |
| `rapid-arm-emu` | rapid-arm-emu | `d762465dfd05` | GPL-3.0-only (no "or later" grant) | A64 decoder and software MMU, for a later x86-64 JIT |

Rules:

- Cite what you port: `// Ported from Eden src/core/hle/service/hle_ipc.cpp @ cf654052 (GPL-3.0-or-later)`.
- Reference paths in code, docs, issues and PRs are relative to the reference folder
  (`eden/src/...`), never absolute paths on someone's machine.
- Update the commit here and in `tools/fetch-refs.sh` together.
- `rapid-arm-emu`'s decoder is 228k lines of generated Rust and slow to compile. Don't vendor it
  for the load-time patcher, which only needs to recognise a handful of encodings; revisit when the
  x86-64 JIT starts.

## Documentation sources

A few layout facts are in none of the pinned sources above. They come from the SwitchBrew wiki,
which is not a git repository, so it cannot be pinned to a commit. The pages were read on
2026-11-11; no revision id is recorded because the ids the wiki reported on two lookups did not
agree. Treat these as third-party documentation, weaker than the pinned sources: nothing here has
been cross-checked against a file built from homebrew.

| Page | Used for | Status |
|---|---|---|
| NSO (`switchbrew.org/wiki/NSO`) | Hash-check flag bits 3 to 5 (hash over the decompressed segment); the last words of the first two segment headers are the module name offset and size | Documented, unverified: no fixture exercises either (`crates/loader/src/nso.rs`) |
| NPDM (`switchbrew.org/wiki/NPDM`) | META words 0x10 (Switch 2 remote memory size), 0x14 (system resource size), 0x18 (version); that service names support a `*` wildcard | Offsets agree with Eden. The trailing-only, prefix-match reading of `*` is a **guess**: the page does not say it (`crates/loader/src/npdm/service_access.rs`) |

Rules:

- Code that depends on a fact from these pages says so next to the fact, and says what would
  confirm it.
- A rule that appears in no source at all (for example how an `ACI0` is checked against its
  `ACID`) is a guess, stated as one in the module docs, not a documentation-sourced fact.
