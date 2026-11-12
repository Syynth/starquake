#!/bin/bash
# Usage: run-nro.sh <nro> <run-dir> [seconds]
# Needs STARQUAKE_ORACLE: the folder that holds the reference emulator's build
# (docs/reference-emulator.md).
# Runs eden-cli (Eden-NX2 build) with SVC+IPC tracing, the null renderer, a hidden window and
# single-core mode. The portable user directory is <run-dir>/user (eden-cli uses ./user when it
# exists); HOME and XDG_* also point inside <run-dir>, so nothing touches the real config.
# eden-cli's exit callback is a no-op, so it never quits when the guest exits: the run is killed
# after [seconds] (default 6). Writes <run-dir>/stdout.log, end.txt and the extracted traces.
set -u
ORACLE=${STARQUAKE_ORACLE:?set STARQUAKE_ORACLE to the folder holding build-nx2}
NRO=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
RUN=$2; SECS=${3:-6}
rm -rf "$RUN"; mkdir -p "$RUN/user" "$RUN/home"
cd "$RUN"
START=$(date +%s)
HOME="$RUN/home" XDG_DATA_HOME="$RUN/home/data" XDG_CONFIG_HOME="$RUN/home/config" \
XDG_CACHE_HOME="$RUN/home/cache" \
EDEN_NX2_SVC_TRACE=1 EDEN_NX2_IPC_TRACE=1 EDEN_NX2_HIDE_WINDOW=1 SDL_MAC_BACKGROUND_APP=1 \
  nice "$ORACLE/build-nx2/bin/eden-cli" -n -s --filter "${EDEN_FILTER:-*:Info Core.ARM:Critical}" -g "$NRO" > stdout.log 2>&1 &
PID=$!
for i in $(seq 1 $((SECS*10))); do kill -0 $PID 2>/dev/null || break; sleep 0.1; done
if kill -0 $PID 2>/dev/null; then
  kill -TERM $PID; sleep 1; kill -KILL $PID 2>/dev/null; echo "killed by script after ${SECS}s" > end.txt
else
  wait $PID; echo "eden-cli exited by itself rc=$? after $(( $(date +%s) - START ))s" > end.txt
fi
# Strip ANSI colours, then pull out the interesting streams.
tr -d '\000' < stdout.log | sed $'s/\x1b\\[[0-9;]*m//g' > log.txt
# (grep -a: the log contains a few non-text bytes, which make plain grep go quiet.)
grep -a 'Kernel.SVC.*svc 0x' log.txt > svc.txt
grep -aE 'Service.*: (ipc|tipc) ' log.txt > ipc.txt
grep -a 'Debug.Emulated' log.txt > guest.txt
grep -aE 'svc 0x|: ipc |: tipc |Debug.Emulated|nimplemented|exiting|ExceptionRaised|Critical|NSException' log.txt > timeline.txt
gzip -f log.txt; rm -f stdout.log
cat end.txt
