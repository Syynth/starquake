#!/bin/sh
# Build, sign with the hypervisor entitlement, run.
set -e
cd "$(dirname "$0")"
cargo build --release --quiet -p hvprobe
codesign --sign - --entitlements entitlements.plist --force ../../target/release/hvprobe 2>/dev/null
exec ../../target/release/hvprobe
