#!/bin/sh
# Cargo runner for macOS on Apple Silicon (see .cargo/config.toml).
#
# Hypervisor.framework refuses to create a VM unless the calling binary is signed with the
# com.apple.security.hypervisor entitlement. This ad-hoc signs each binary once, then runs it, so
# `cargo run`, `cargo test` and `cargo nextest run` work for code that uses the hypervisor.
#
# "Once" is decided by what the file is, not by when it was signed. The marker next to a binary
# records the inode, modification time and size of the file that was signed, and the binary is
# signed again whenever those no longer match. A time comparison is not enough: before every
# `cargo run`, cargo replaces target/<profile>/<name> with a fresh copy of the unsigned binary it
# built earlier, which keeps its old modification time and so looks older than any marker. The
# inode is there because time (whole seconds) and size alone do not tell a new file from the one
# that was signed. xtask/tests/macos_runner.rs replays both cases.
#
# Test runners start the same binary several times at once, and codesign replaces the file while
# it works, so signing is done under a lock.
set -e
bin="$1"
marker="$bin.hvsigned"

# Prints what identifies the file at $bin right now. The system's own stat is named in full: with
# GNU coreutils first on PATH, `stat -f` reports the file system instead of the file.
identity() {
    /usr/bin/stat -f '%i %m %z' "$bin"
}

# Succeeds if the file at $bin is the one this script last signed.
is_signed() {
    [ -f "$marker" ] && [ "$(cat "$marker")" = "$(identity)" ]
}

if ! is_signed; then
    lock="$bin.hvlock"
    tries=0
    until mkdir "$lock" 2>/dev/null; do
        tries=$((tries + 1))
        # A lock older than ~10 s was left by a killed run.
        if [ "$tries" -gt 200 ]; then rmdir "$lock" 2>/dev/null || true; tries=0; fi
        sleep 0.05
    done
    if ! is_signed; then
        entitlements="$(dirname "$0")/hypervisor.entitlements"
        if ! codesign --sign - --entitlements "$entitlements" --force "$bin" 2>/dev/null; then
            rmdir "$lock"
            echo "macos-runner: codesign failed for $bin" >&2
            exit 1
        fi
        identity > "$marker"
    fi
    rmdir "$lock"
fi
exec "$@"
