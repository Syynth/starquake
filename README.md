# starquake

An experimental Nintendo Switch 2 emulator in Rust, for macOS (Apple Silicon) and Linux.
Early days: homebrew programs built with libnx run natively under Apple's hypervisor, across
six emulated cores, and their system calls and service requests match a reference emulator's
traces. No commercial game runs yet, and nothing is drawn: the graphics half is designed and its
first pieces are being built. Starquake is not affiliated with Nintendo and ships no keys, firmware or games.

- Architecture: [docs/architecture.md](docs/architecture.md)
- Why and how: [docs/decision-log.md](docs/decision-log.md),
  [docs/switch2-eden-diff-scoping.md](docs/switch2-eden-diff-scoping.md)
- Check everything: `cargo xtask ci`
- After cloning, once: `just setup` (installs the Conventional Commits `commit-msg` hook)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

Licensed under GPL-3.0-or-later.
