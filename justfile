# The gate: fmt check, clippy -D warnings, tests, doctests. CI runs exactly this.
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
