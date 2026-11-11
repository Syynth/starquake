#!/bin/sh
# Cargo runner for macOS on Apple Silicon (see .cargo/config.toml).
#
# Hypervisor.framework refuses to create a VM unless the calling binary is signed with the
# com.apple.security.hypervisor entitlement. This ad-hoc signs each binary once, then runs it, so
# `cargo run`, `cargo test` and `cargo nextest run` work for code that uses the hypervisor.
#
# Test runners start the same binary several times at once, and codesign replaces the file while
# it works, so signing is done under a lock.
set -e
bin="$1"
marker="$bin.hvsigned"
if [ ! "$marker" -nt "$bin" ]; then
    lock="$bin.hvlock"
    tries=0
    until mkdir "$lock" 2>/dev/null; do
        tries=$((tries + 1))
        # A lock older than ~10 s was left by a killed run.
        if [ "$tries" -gt 200 ]; then rmdir "$lock" 2>/dev/null || true; tries=0; fi
        sleep 0.05
    done
    if [ ! "$marker" -nt "$bin" ]; then
        entitlements="$(dirname "$0")/hypervisor.entitlements"
        if ! codesign --sign - --entitlements "$entitlements" --force "$bin" 2>/dev/null; then
            rmdir "$lock"
            echo "macos-runner: codesign failed for $bin" >&2
            exit 1
        fi
        touch "$marker"
    fi
    rmdir "$lock"
fi
exec "$@"
