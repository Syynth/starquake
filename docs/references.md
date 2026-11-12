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
| `atmosphere` | Atmosphère (`Atmosphere-NX/Atmosphere`, branch `master`) | `4e3f37038d9e` | GPL-2.0-only | **Facts only**: result codes (`libraries/libvapours/include/vapours/results/`), kernel behaviour (`libraries/libmesosphere`), system-module command ids and layouts (`stratosphere`). Never ported from |
| `rapid-arm-emu` | rapid-arm-emu | `d762465dfd05` | GPL-3.0-only (no "or later" grant) | A64 decoder and software MMU, for a later x86-64 JIT |
| `nup` | NUP (`a55uka/nup`, branch `meow`) | `a4840f06c94d` | MIT (`LICENSE`, Copyright (c) 2026 a55uka) | The NUP container for decrypted dumps: `SPEC.md` (version 1.0-draft) and the reference tool's reader, verifier and packer (`tools/nup/nup/`), for `crates/loader/src/nup.rs` |

`tools/fetch-refs.sh` never prompts for credentials (every source is fetched anonymously), stops at
the first source it cannot fetch with a message naming it, and does not touch the network for a
folder that is already at its pinned commit.

Rules:

- **Atmosphère is for facts only.** Its licence is GPL-2.0-only (its README says GPLv2; we found
  no "or later" grant), which cannot be combined with this project's GPL-3.0-or-later, so none of
  its code is ported, translated or paraphrased closely. Read it to settle a fact (a result code's
  value, a command id, a struct layout), write the code from the fact, and cite the file in the
  comment: `// Value from atmosphere/libraries/libvapours/include/vapours/results/<file> @ 4e3f3703 (fact only)`.
  A comment that cites it says "fact only". Bring the fact, not the expression.
- **libnx is ISC**, a permissive licence: code may be ported from it, but the ISC notice from
  `libnx/LICENSE.md` (copyright line and permission text) travels with the port, in the file
  header next to the citation. The source files carry only `@copyright libnx Authors`; the
  permission text is in that file. (`CLAUDE.md`, "Licensing", has the whole list.)
- Cite what you port: `// Ported from Eden src/core/hle/service/hle_ipc.cpp @ cf654052 (GPL-3.0-or-later)`.
- Reference paths in code, docs, issues and PRs are relative to the reference folder
  (`eden/src/...`), never absolute paths on someone's machine.
- Update the commit here and in `tools/fetch-refs.sh` together.
- **NUP: the spec wins over the tool.** `nup/SPEC.md` is the format's definition and the reader
  follows it; the Python tool settles what the spec leaves open. Where they disagree the reader
  follows the spec and says so: at the pinned commit the tool numbers the meta block's TLV types
  from 0 (`format.py`, `class TVL`) while the spec numbers them from 1 (§5).
- `rapid-arm-emu`'s decoder is 228k lines of generated Rust and slow to compile. Don't vendor it
  for the load-time patcher, which only needs to recognise a handful of encodings; revisit when the
  x86-64 JIT starts.

## Documentation sources

The **Arm Architecture Reference Manual for A-profile architecture (Arm DDI 0487)** is the source
that comments in `crates/memory` and `crates/cpu` cite for descriptor bits, system registers and
`ESR_EL1` layouts. It is not a git repository, and Arm distributes it as PDF and HTML only
(`https://developer.arm.com/documentation/ddi0487/latest`), so it cannot be pinned to a commit and
is not fetched. The issue letter is not recorded in the existing citations. Cite it as the code does:
`Arm DDI 0487`, then the section **by name** (the numbering moves between issues), and mark a
bit position you did not check against the manual as unchecked. Treat a layout taken from memory
as a guess until a vCPU run (`tools/hvprobe`) or the manual confirms it.

A few layout facts are in none of the pinned sources above. They come from the SwitchBrew wiki,
which is not a git repository, so it cannot be pinned to a commit. The pages were read on
2026-11-11; no revision id is recorded because the ids the wiki reported on two lookups did not
agree. Treat these as third-party documentation, weaker than the pinned sources. Where a claim
has since been checked against Atmosphère (facts only) or the homebrew fixture
(`tests/homebrew/hello`, below), the status says so.

| Page | Used for | Status |
|---|---|---|
| NSO (`switchbrew.org/wiki/NSO`) | Hash-check flag bits 3 to 5 (hash over the decompressed segment); the last words of the first two segment headers are the module name offset and size | Hash bits **confirmed** by Atmosphère's loader and the fixture. The module-name reading is **not used**: `elf2nso` writes 1 to both words, and the name is read from the start of rodata instead (`crates/loader/src/nso.rs`) |
| NPDM (`switchbrew.org/wiki/NPDM`) | META words 0x10 (Switch 2 remote memory size), 0x14 (system resource size), 0x18 (version); that service names support a `*` wildcard | Offsets agree with Eden. The trailing-`*`, prefix-match reading is **confirmed** by Atmosphère's sm and the fixture (`crates/loader/src/npdm/service_access.rs`) |

Rules:

- Code that depends on a fact from these pages says so next to the fact, and says what would
  confirm it.
- A rule that appears in no source at all (for example how a Switch 2 NPDM is checked against
  its `ACID`: no Switch 2 source does) is a guess, stated as one in the module docs, not a
  documentation-sourced fact. A rule Atmosphère implements differently is a listed divergence
  instead (for example file permissions: fs intersects the `ACI0` and `ACID` bits and never
  refuses; see the table in `crates/loader/src/npdm/validate.rs`).

## Homebrew toolchain

The fixtures in `tests/homebrew/` are built with devkitPro's devkitA64 and libnx, and
switch-tools' `elf2nro`, `elf2nso` and `npdmtool`; `tests/homebrew/README.md` says how. The gate
does not need them. On macOS, `tools/setup-devkitpro.sh` installs them into `/opt/devkitpro`: it
downloads devkitPro's installer package from its GitHub release (pinned in the script), asks
for your password once, installs the `switch-dev` package group and adds `DEVKITPRO` and
`DEVKITA64` to `~/.zshrc`. The versions a fixture was built with are in its `manifest.toml`.

These tools are not reference sources: a fact settled only by a fixture is settled for what
switch-tools writes, and is cited that way.
