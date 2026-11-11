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
| `rapid-arm-emu` | rapid-arm-emu | `d762465dfd05` | GPL-3.0-only (no "or later" grant) | A64 decoder and software MMU, for a later x86-64 JIT |

Rules:

- Cite what you port: `// Ported from Eden src/core/hle/service/hle_ipc.cpp @ cf654052 (GPL-3.0-or-later)`.
- Reference paths in code, docs, issues and PRs are relative to the reference folder
  (`eden/src/...`), never absolute paths on someone's machine.
- Update the commit here and in `tools/fetch-refs.sh` together.
- `rapid-arm-emu`'s decoder is 228k lines of generated Rust and slow to compile. Don't vendor it
  for the load-time patcher, which only needs to recognise a handful of encodings; revisit when the
  x86-64 JIT starts.
