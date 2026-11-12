#!/bin/sh
# Fetches the reference sources at the commits pinned in docs/references.md.
# Read-only material for porting; never built, never copied into this repository wholesale.
#
# Never prompts: every source is public and fetched anonymously, and an agent or a CI job cannot
# answer a username prompt. Prompting is switched off for every git this script starts, whatever
# the caller's environment says, and a source that cannot be fetched ends the script with a
# message naming it.
set -e

GIT_TERMINAL_PROMPT=0
GIT_ASKPASS=false
SSH_ASKPASS=false
GCM_INTERACTIVE=never
GIT_LFS_SKIP_SMUDGE=1 # rapid-arm-emu has LFS files; the pointers are enough
export GIT_TERMINAL_PROMPT GIT_ASKPASS SSH_ASKPASS GCM_INTERACTIVE GIT_LFS_SKIP_SMUDGE

refs="$("$(dirname "$0")/refs-path.sh")"
mkdir -p "$refs"
cd "$refs"

give_up() { # dir url commit
    echo "fetch-refs: could not fetch $1 ($2) at $3. This script never asks for a username or password: check the network, the address and the commit, and that the source is public." >&2
    exit 1
}

# Whether the checkout in $1 is already at commit $2, so that it needs no network (and still
# works when the source has moved or gone away).
at_commit() { # dir commit
    [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" = "$2" ]
}

fetch_commit_steps() { # dir url commit
    if [ ! -d "$1/.git" ] && [ ! -f "$1/.git" ]; then
        git init -q "$1" || return 1
        git -C "$1" remote add origin "$2" || return 1
    elif at_commit "$1" "$3"; then
        return 0
    fi
    git -C "$1" fetch -q --depth 1 origin "$3" || return 1
    git -C "$1" checkout -q --detach FETCH_HEAD
}

fetch_commit() { # dir url commit
    fetch_commit_steps "$@" || give_up "$1" "$2" "$3"
}

mesa_url=https://gitlab.freedesktop.org/mesa/mesa.git
mesa_commit=52238762dd44939c2912185f4df00b90d8306225

fetch_mesa_steps() {
    if [ -d mesa-nouveau/.git ] && at_commit mesa-nouveau "$mesa_commit"; then
        return 0
    fi
    if [ ! -d mesa-nouveau/.git ]; then
        git clone -q --depth 1 --filter=blob:none --sparse "$mesa_url" mesa-nouveau || return 1
    fi
    git -C mesa-nouveau sparse-checkout set src/nouveau/compiler src/nouveau/headers src/nouveau/nil src/nouveau/mme || return 1
    git -C mesa-nouveau fetch -q --depth 1 --filter=blob:none origin "$mesa_commit" || return 1
    git -C mesa-nouveau checkout -q --detach FETCH_HEAD
}

fetch_commit eden https://git.eden-emu.dev/eden-emu/eden.git cf654052c416f9e7c80cef0a7d321476b1c52f99
fetch_commit eden-nx2 https://git.eden-emu.dev/Takoroka/eden-nx2.git 50fa86bbbc403ad565eb44fcc281f16f77c69e15
fetch_commit eden-pr4588 https://git.eden-emu.dev/ponting/eden.git e5fb3b97c94bab1e4fd19dc576aa96195d417182
fetch_commit libnx https://github.com/switchbrew/libnx.git b156b4774fb2df7b93402169509266da7547fc4c
# GPL-2.0-only: a reference for facts (result codes, ids, layouts), never for code.
fetch_commit atmosphere https://github.com/Atmosphere-NX/Atmosphere.git 4e3f37038d9e82dde7054897213ab32b2ecf651f

# MIT: the NUP container's specification and reference tool (branch `meow`).
fetch_commit nup https://github.com/a55uka/nup.git a4840f06c94ddeb414eea5b33c9704ffe995f11c

fetch_mesa_steps || give_up mesa-nouveau "$mesa_url" "$mesa_commit"

fetch_commit rapid-arm-emu https://github.com/vrtgs/rapid-arm-emu.git d762465dfd0579acba81a8dc944296522a61f929

echo "references are in $refs"
