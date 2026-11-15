# Licences of ported code

Starquake is licensed under the GNU General Public License, version 3 or (at your option) any
later version: the text is in `LICENSE` at the repository root.

Some files hold code ported from other projects under licences that are compatible with it. Each
such file names its source, licence and copyright holders in its header, and the licence texts
are here so they travel with the code:

| File | Used for |
|---|---|
| `GPL-2.0-or-later.txt` | Code from yuzu and early Eden under GPL-2.0-or-later (some host shaders) |
| `Apache-2.0.txt` | Code derived from FasTC, through Eden's ASTC decoder |
| `MIT.txt` | Code ported from Mesa (nouveau, NAK, nil). The copyright line is a template: each ported file's header names the holders |
| `ISC.txt` | Code ported from libnx; the text with libnx's own copyright line, as `CLAUDE.md` requires |

Code ported from GPL-3.0-or-later projects (Eden, oboromi, rapid-arm-emu) is covered by `LICENSE`.
Atmosphère is GPL-2.0-only and is used for facts only, never ported.
