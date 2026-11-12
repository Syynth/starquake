# starquake

An experimental Nintendo Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux.
Early days: what runs is a walking skeleton, small synthetic test programs executed natively
under Apple's hypervisor with their system calls traced
([docs/architecture.md](docs/architecture.md), "What runs today"). No game or homebrew runs
yet. Starquake is not affiliated with Nintendo and ships no keys, firmware or games.

- Architecture: [docs/architecture.md](docs/architecture.md)
- Why and how: [docs/decision-log.md](docs/decision-log.md),
  [docs/switch2-eden-diff-scoping.md](docs/switch2-eden-diff-scoping.md)
- Check everything: `cargo xtask ci`
- After cloning, once: `just setup` (installs the Conventional Commits `commit-msg` hook)

Licensed under GPL-3.0-or-later.
