# The startup fixture as a reference emulator runs it

`startup.trace` is what `../startup.nro` does at the system boundary from its first instruction
to the end of its process, recorded from a reference emulator and normalised so that another
emulator's run can be compared with it. The format is the hello fixture's
(`../../hello/reference/README.md`, "Format"), with two additions this fixture needed:

- `placement<n>+<offset>`: an address in a range libnx placed at a random page of the ASLR
  region (the shared memories of `hid` and `time:u`). Placements are numbered in the order the
  trace maps them, and anchored at the page libnx probes with `QueryMemory` before mapping, a
  guard page below the range (`tools/oracle/normalise.py` cites the libnx source). The script
  refuses a trace with a placement where libnx would not map one; its module docs give libnx's
  conditions and the two cases where it is stricter, which these traces do not reach.
- `out=random`: `GetInfo`'s answer for the random entropy (info type `0xb`), which differs from
  run to run.

`crates/core/tests/homebrew.rs` runs the fixture in starquake and compares its trace with this
one, line by line, as far as it matches; the test fails if that is fewer lines than the number
it records (`STARTUP_MATCHED`), so the number only goes up.

## Provenance

| | |
|---|---|
| Emulator | Eden-NX2 at commit `50fa86bbbc40` (`eden-nx2` in `docs/references.md`), built and run as `docs/reference-emulator.md` says, with the one patch it names (`tools/oracle/eden-nx2-oracle.patch`: a window-title fix, nothing the guest can see) |
| Run | `tools/oracle/run-nro.sh` (null renderer, one core, the emulator's own system-call and IPC trace switches, default settings otherwise: a 4 GiB console), then `tools/oracle/summarize.py` with the stop pattern below, whose `startup.txt` is what `tools/oracle/normalise.py` reads |
| Program | `../startup.nro`, SHA-256 `57b4200993a45b59156a447a0fe6c23aedf7a684ca1c48b74f4f698520ba80d2` (the hash `../manifest.toml` records) |
| Recorded | 2026-11-12, three runs |

The three runs differ only in addresses, timestamps and the random entropy, and normalise to
the same text, byte for byte. The raw run folders are not in the repository
(`docs/reference-emulator.md`, "What may be committed").

## How the run ends

The program never calls `ExitProcess`. Started without a homebrew loader, libnx ends it through
`appletOE` (`_appletCleanup` in `libnx/nx/source/services/applet.c` @ b156b477: with no loader
environment, `envIsNso()` holds and an application calls `ISelfController::Exit`). Eden's
`ISelfController::Exit` terminates the process (`eden/src/core/hle/service/am/service/self_controller.cpp`),
so the `SendSyncRequest` carrying it returns `0x7601`, the kernel's `ResultTerminationRequested`
(`eden/src/core/hle/kernel/svc_results.h`, module 1, description 59), and nothing follows. That
is the trace's last line. On a console the request would not return to the guest at all; the
result is what the reference's thread saw while it was being terminated, and starquake need not
match it.

The other notes in `docs/reference-emulator.md` apply here too: every `SendSyncRequest` with
result `0xf601` is the reference's answer to a session's close request, which is unverified.

## What starquake lacks

Every system call and service request in the trace that starquake does not implement yet, once
each, in the order of first use. "Line" is the line of `startup.trace`, counting its comment
lines from 1. A call is listed when its status in `starquake_kernel::svc::SvcId` is not
`Implemented`; a request when it is not a control command the service framework answers
(`IpcController`) and not a command of a service starquake has (`sm:`, `set:sys`) under the same
name. The result is the reference's: a call's own result, or for a request the result of the
`SendSyncRequest` that carried it (the trace does not record a reply's own result code).
Requests to a sub-interface (`IApplicationProxy`, `ISelfController`, ...) are named by the
interface, as the reference logs them.

`crates/core/tests/homebrew.rs` writes this list from the trace and starquake's tables, and
fails (`the_startup_gap_list_in_its_readme_is_the_one_the_trace_gives`) when the text between
the two markers below differs; its failure message holds the text to put there.

<!-- gap list: written by crates/core/tests/homebrew.rs -->
| Line | Call or request | Reference's result |
|---:|---|---|
| 39 | `ipc apm::0 (OpenSession)` | `0x0` |
| 47 | `ipc appletOE::0 (OpenApplicationProxy)` | `0x0` |
| 49 | `ipc IApplicationProxy::20 (GetApplicationFunctions)` | `0x0` |
| 51 | `ipc IApplicationProxy::11 (GetLibraryAppletCreator)` | `0x0` |
| 53 | `ipc IApplicationProxy::0 (GetCommonStateGetter)` | `0x0` |
| 55 | `ipc IApplicationProxy::1 (GetSelfController)` | `0x0` |
| 57 | `ipc IApplicationProxy::2 (GetWindowController)` | `0x0` |
| 59 | `ipc IWindowController::1 (GetAppletResourceUserId)` | `0x0` |
| 61 | `ipc IApplicationProxy::3 (GetAudioController)` | `0x0` |
| 63 | `ipc IApplicationProxy::4 (GetDisplayController)` | `0x0` |
| 65 | `ipc IApplicationProxy::1000 (GetDebugFunctions)` | `0x0` |
| 67 | `ipc ISelfController::91 (GetAccumulatedSuspendedTickChangedEvent)` | `0x0` |
| 69 | `ipc ICommonStateGetter::0 (GetEventHandle)` | `0x0` |
| 71 | `ipc ICommonStateGetter::9 (GetCurrentFocusState)` | `0x0` |
| 73 | `ipc IWindowController::10 (AcquireForegroundRights)` | `0x0` |
| 75 | `ipc ISelfController::13 (SetFocusHandlingMode)` | `0x0` |
| 77 | `ipc ISelfController::16 (SetOutOfFocusSuspendingEnabled)` | `0x0` |
| 79 | `ipc IApplicationFunctions::40 (NotifyRunning)` | `0x0` |
| 81 | `ipc ICommonStateGetter::5 (GetOperationMode)` | `0x0` |
| 83 | `ipc ICommonStateGetter::6 (GetPerformanceMode)` | `0x0` |
| 85 | `ipc ISelfController::11 (SetOperationModeChangedNotification)` | `0x0` |
| 87 | `ipc ISelfController::12 (SetPerformanceModeChangedNotification)` | `0x0` |
| 93 | `ipc hid::0 (CreateAppletResource)` | `0x0` |
| 95 | `ipc IAppletResource::0 (GetSharedMemoryHandle)` | `0x0` |
| 102 | `svc 0x13 MapSharedMemory` | `0x0` |
| 107 | `ipc time:u::0 (GetStandardUserSystemClock)` | `0x0` |
| 109 | `ipc time:u::1 (GetStandardNetworkSystemClock)` | `0x0` |
| 111 | `ipc time:u::2 (GetStandardSteadyClock)` | `0x0` |
| 113 | `ipc time:u::3 (GetTimeZoneService)` | `0x0` |
| 115 | `ipc time:u::4 (GetStandardLocalSystemClock)` | `0x0` |
| 117 | `ipc time:u::20 (GetSharedMemoryNativeHandle)` | `0x0` |
| 121 | `ipc ITimeZoneService::101 (ToCalendarTimeWithMyRule)` | `0x0` |
| 129 | `ipc fsp-srv::1 (SetCurrentProcess)` | `0x0` |
| 135 | `ipc fsp-srv::18 (OpenSdCardFileSystem)` | `0x0` |
| 141 | `ipc vi:m::2 (GetDisplayService)` | `0x0` |
| 143 | `ipc IApplicationDisplayService::100 (GetRelayService)` | `0x0` |
| 145 | `ipc IApplicationDisplayService::101 (GetSystemDisplayService)` | `0x0` |
| 147 | `ipc IApplicationDisplayService::102 (GetManagerDisplayService)` | `0x0` |
| 149 | `ipc IApplicationDisplayService::103 (GetIndirectDisplayTransactionService)` | `0x0` |
| 151 | `ipc IApplicationDisplayService::1010 (OpenDisplay)` | `0x0` |
| 153 | `ipc ISelfController::40 (CreateManagedDisplayLayer)` | `0x0` |
| 155 | `ipc IApplicationDisplayService::2020 (OpenLayer)` | `0x0` |
| 157 | `ipc IApplicationDisplayService::2101 (SetLayerScalingMode)` | `0x0` |
| 159 | `ipc IHOSBinderDriver::1 (AdjustRefcount)` | `0x0` |
| 163 | `ipc IHOSBinderDriver::2 (GetNativeHandle)` | `0x0` |
| 165 | `ipc IHOSBinderDriver::3 (TransactParcelAuto)` | `0x0` |
| 171 | `svc 0x15 CreateTransferMemory` | `0x0` |
| 172 | `ipc nvdrv::3 (Initialize)` | `0x0` |
| 177 | `ipc nvdrv::8 (SetAruid)` | `0x0` |
| 179 | `ipc nvdrv::0 (Open)` | `0x0` |
| 183 | `ipc nvdrv::1 (Ioctl)` | `0x0` |
| 204 | `ipc nvdrv::2 (Close)` | `0x0` |
| 218 | `ipc IApplicationDisplayService::2021 (CloseLayer)` | `0x0` |
| 220 | `ipc IApplicationDisplayService::1020 (CloseDisplay)` | `0x0` |
| 241 | `svc 0x14 UnmapSharedMemory` | `0x0` |
| 267 | `ipc ISelfController::0 (Exit)` | `0x7601` |
<!-- end of the gap list -->

## Regenerating

```sh
STARQUAKE_ORACLE=<folder> tools/oracle/run-nro.sh tests/homebrew/startup/startup.nro <run-folder> 8
python3 tools/oracle/summarize.py <run-folder> 'svc 0x21 out .*\[0\]=0x7601'
tools/oracle/normalise.py --nro tests/homebrew/startup/startup.nro <run-folder>/startup.txt > normalised.txt
```

and put the file's five header lines back in front. Then update the gap list as above.
