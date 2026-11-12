# The reference emulator

Starquake's behaviour is compared with a locally built copy of Eden-NX2 (`docs/references.md`,
`eden-nx2`, commit `50fa86bbbc40`). It runs the same homebrew files headlessly and logs every
system call and service request, which gives an ordered trace to match. The first use is the
libnx fixture, `tests/homebrew/hello/hello.nro`.

It is a tool on the developer's machine, never a dependency: nothing in the build or the gate
needs it, and CI never runs it.

## What it is trusted for

- **The order of system calls and service requests** a program makes, and their arguments.
- **Result codes**, with care. The reference is itself an emulator: a result it returns is its
  authors' reading of Horizon, not a measurement of a console.

Known places where the reference is wrong or unverified, which starquake does not copy:

- Its `ExitProcess` returns to the guest. Starquake ends the run.
- It answers a session's close request with `0xf601`. This is unverified against a console.
  Starquake returns the same, for a reason of its own: a console's server framework closes its
  end of the session on a close request without replying (fact only:
  `atmosphere/libraries/libstratosphere/source/sf/hipc/sf_hipc_server_session_manager.cpp`,
  `ProcessRequest`), and the kernel ends a request whose server end closed with
  `ResultSessionClosed`, `0xf601` (`eden/src/core/hle/kernel/k_server_session.cpp`,
  `CleanupRequests`). Both are read from sources, not measured.

Add to this list whenever a comparison turns up another.

## Building it (macOS on Apple Silicon)

The build lives outside the repository, in a folder named by `STARQUAKE_ORACLE`. It takes about
20 minutes and needs no keys and no firmware.

1. Copy the pinned `eden-nx2` reference source into `$STARQUAKE_ORACLE/eden-nx2-src`.
2. Apply `tools/oracle/eden-nx2-oracle.patch` (a change to Eden-NX2's source, so GPL-3.0-or-later
   like the file it patches). Without it the command-line frontend aborts two
   seconds into every run on macOS, because it sets the window title from a timer thread.
3. Configure a `Release` build into `$STARQUAKE_ORACLE/build-nx2` with the command-line frontend
   only: `ENABLE_QT=OFF`, `YUZU_CMD=ON`, and `ENABLE_WEB_SERVICE`, `ENABLE_CUBEB`, `ENABLE_LIBUSB`
   and `ENABLE_UPDATE_CHECKER` off. The exact configure line of the first build was not kept;
   record it here the next time the build is made.
4. `ninja -C build-nx2 eden-cli`.

## Running it

```sh
STARQUAKE_ORACLE=<folder> tools/oracle/run-nro.sh tests/homebrew/hello/hello.nro <run-folder>
python3 tools/oracle/summarize.py <run-folder>
```

`run-nro.sh` runs with the null renderer, a hidden window and one core, keeps the emulator's
settings inside the run folder, and stops it after six seconds (the frontend never quits by
itself). It leaves `svc.txt`, `ipc.txt`, `guest.txt` and the full log. `summarize.py` writes
`startup.txt`, the calls up to the first `ExitProcess`, and prints a count per system call.

Three runs of the fixture were identical apart from addresses and timestamps.

A program that keeps libnx's own start-up (`tests/homebrew/startup`) never calls `ExitProcess`
there: started without a homebrew loader, libnx ends it through `appletOE`'s
`ISelfController::Exit`, after which the emulator stops the process and its `SendSyncRequest`
returns `0x7601`. Pass `summarize.py` a stop pattern for that line, as the fixture's README says.

## What may be committed

- A **normalised** trace of a homebrew program (addresses and timestamps masked), with a note
  saying which reference commit and which program produced it. The fixtures' are
  `tests/homebrew/hello/reference/startup.trace` and
  `tests/homebrew/startup/reference/startup.trace`, made from a run's `startup.txt` by
  `tools/oracle/normalise.py`; the README beside the first has the format.
- Never a raw run folder, and never anything from a run of a commercial title
  (`docs/specs/observability.md`, "Storage and what may be committed").
