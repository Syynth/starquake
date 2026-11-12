#!/bin/sh
# Fetches the reference sources at the commits pinned in docs/references.md.
# Read-only material for porting; never built, never copied into this repository wholesale.
set -e
refs="$("$(dirname "$0")/refs-path.sh")"
mkdir -p "$refs"
cd "$refs"

fetch_commit() { # dir url commit
    if [ ! -d "$1/.git" ] && [ ! -f "$1/.git" ]; then
        git init -q "$1"
        git -C "$1" remote add origin "$2"
    fi
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q --detach FETCH_HEAD
}

fetch_commit eden https://git.eden-emu.dev/eden-emu/eden.git cf654052c416f9e7c80cef0a7d321476b1c52f99
fetch_commit eden-nx2 https://git.eden-emu.dev/Takoroka/eden-nx2.git 50fa86bbbc403ad565eb44fcc281f16f77c69e15
fetch_commit eden-pr4588 https://git.eden-emu.dev/ponting/eden.git e5fb3b97c94bab1e4fd19dc576aa96195d417182
fetch_commit libnx https://github.com/switchbrew/libnx.git b156b4774fb2df7b93402169509266da7547fc4c

if [ ! -d mesa-nouveau/.git ]; then
    git clone -q --depth 1 --filter=blob:none --sparse https://gitlab.freedesktop.org/mesa/mesa.git mesa-nouveau
fi
git -C mesa-nouveau sparse-checkout set src/nouveau/compiler src/nouveau/headers src/nouveau/nil src/nouveau/mme
git -C mesa-nouveau fetch -q --depth 1 --filter=blob:none origin 52238762dd44939c2912185f4df00b90d8306225
git -C mesa-nouveau checkout -q --detach FETCH_HEAD

GIT_LFS_SKIP_SMUDGE=1 fetch_commit rapid-arm-emu https://github.com/vrtgs/rapid-arm-emu.git d762465dfd0579acba81a8dc944296522a61f929

echo "references are in $refs"
