#!/bin/sh
# Prints the folder holding the pinned reference sources (see docs/references.md).
# It is a sibling of the main checkout, shared by every worktree; STARQUAKE_REFS overrides it.
set -e
if [ -n "$STARQUAKE_REFS" ]; then
    echo "$STARQUAKE_REFS"
    exit 0
fi
common="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
echo "$(dirname "$(dirname "$common")")/starquake-refs"
