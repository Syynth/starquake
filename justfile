# The gate: fmt check, unused-dependency checks, cargo deny, clippy -D warnings, tests, doctests.
# CI runs exactly this. `cargo xtask ci --offline` skips the advisory check when there is no network.
ci:
    cargo xtask ci

# Auto-fix what can be fixed
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-dirty

# Fetch the pinned reference sources into ../starquake-refs
refs:
    tools/fetch-refs.sh

# Measure Hypervisor.framework behaviour (macOS on Apple Silicon only)
hvprobe:
    cargo run --release -p hvprobe

# One-time setup after cloning: use the repo's git hooks (.githooks/commit-msg checks that commit
# messages follow Conventional Commits)
setup:
    git config core.hooksPath .githooks
