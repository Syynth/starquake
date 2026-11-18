//! `appletOE`, the applet manager's service for an application: as far as a libnx start-up
//! asks.
//!
//! Ported from Eden's `am` service @ cf654052 (GPL-3.0-or-later):
//! `eden/src/core/hle/service/am/service/application_proxy_service.cpp`
//! (`IApplicationProxyService`), `application_proxy.cpp` (`IApplicationProxy`),
//! `common_state_getter.cpp` (`ICommonStateGetter`), `self_controller.cpp` (`ISelfController`),
//! `window_controller.cpp` (`IWindowController`), `application_functions.cpp`
//! (`IApplicationFunctions`), `audio_controller.cpp`, `display_controller.cpp`,
//! `library_applet_creator.cpp` and `debug_functions.cpp` (their interfaces' names only), with the
//! applet's state from `eden/src/core/hle/service/am/lifecycle_manager.cpp` and
//! `lifecycle_manager.h` (`LifecycleManager`), `applet.cpp` (`Applet::UpdateSuspensionStateLocked`),
//! `applet_manager.cpp` (the launch), `am_types.h` (`FocusState`, `OperationMode`,
//! `AppletMessage`) and `am_results.h` (`ResultNoMessages`). The request
//! layouts are checked against the client, libnx (`libnx/nx/source/services/applet.c`, ISC).
//! The Switch 2 additions are ported from Eden PR #4588 @ e5fb3b97 (GPL-3.0-or-later):
//! `eden-pr4588/src/core/hle/service/am/service/application_proxy_service.cpp`
//! (`OpenApplicationProxyWithFlags`), `application_proxy.cpp` (`IApplicationProxy` 40 and 60),
//! `performance_functions.cpp` (`IPerformanceFunctions`),
//! `graphics_scaling_parameters_functions.cpp` and `.h` (`IGraphicsScalingParametersFunctions`)
//! and `application_functions.cpp` (`IApplicationFunctions` 320 and 321), with the applet's
//! initial values from `eden-pr4588/src/core/hle/service/am/applet.h` (see "The Switch 2
//! additions"). The other commands a Switch 2 title sends at start-up, and the setters 320
//! reads, are ported from PR #4588 @ e5fb3b97 too, whose handlers for them are Eden's own:
//! `common_state_getter.cpp` (`ICommonStateGetter` 900), `application_functions.cpp`
//! (`IApplicationFunctions` 65, 67, 130 and 330) and `self_controller.cpp` (`ISelfController`
//! 10), with `ScreenshotPermission` and `GamePlayRecordingState` from
//! `eden-pr4588/src/core/hle/service/am/am_types.h` (see "The title's start-up commands" and
//! "Screenshots and recording"). The commands a Switch 2 title sends after those (see "The
//! title's later start-up commands") are ported from Eden @ cf654052, whose handlers Eden-NX2 @
//! 50fa86bb and PR #4588 @ e5fb3b97 share: `application_functions.cpp` (`IApplicationFunctions`
//! 21 and 210), with what 21 reads from
//! `eden/src/core/hle/service/ns/read_only_application_control_data_interface.cpp`
//! (`GetApplicationDesiredLanguage`, `ConvertApplicationLanguageToLanguageCode`),
//! `eden/src/core/hle/service/ns/language.cpp`, `eden/src/core/hle/service/set/settings_types.h`
//! (`LanguageCode`) and `eden/src/common/settings.h` (`language_index`);
//! `common_state_getter.cpp` (`ICommonStateGetter` 66), with
//! `eden/src/core/hle/service/apm/apm_interface.cpp` (`APM_Sys::SetCpuBoostMode`) and
//! `apm_controller.cpp` and `.h` (`Controller::SetFromCpuBoostMode`, `CpuBoostMode`); and
//! `self_controller.cpp` (`ISelfController` 14), with `lifecycle_manager.h`
//! (`SetResumeNotificationEnabled`). `ISelfController` 2400 is ported from PR #4588 @ e5fb3b97
//! alone (`eden-pr4588/src/core/hle/service/am/service/self_controller.cpp`).
//!
//! # Commands
//!
//! Ids and names are Eden's tables. These are exactly the commands the startup fixture's trace
//! sends (`tests/homebrew/startup/reference/startup.trace`; its README's gap list named them),
//! `ISelfController::Exit` (0), which libnx sends when the program ends (see "Where this
//! differs from Eden"), `ISelfController` 2000, a Switch 2 command (see "The graphics
//! server" below), the other Switch 2 additions the scoping doc names (see "The Switch 2
//! additions" below), the commands a Switch 2 title sends at start-up that those leave out
//! (`ICommonStateGetter` 900 and `IApplicationFunctions` 130), `ICommonStateGetter` 1, which a
//! title that waits on the message event sends (see "The messages"), `IApplicationFunctions` 330,
//! next to 320 and 321 in PR's table (see "The title's start-up commands"), and the three
//! commands whose state 320 reads (see "Screenshots and recording"), and the five a Switch 2
//! title sends after those (`IApplicationFunctions` 21 and 210, `ICommonStateGetter` 66,
//! `ISelfController` 14 and 2400; see "The title's later start-up commands"); every other command
//! of these interfaces is left out and answered by the server's unknown-command policy.
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `appletOE` ([`IApplicationProxyService`]) | 0 | `OpenApplicationProxy` | the caller's process id, attached by the kernel; a `u64` placeholder in the payload, not read; copy handle 0: the caller's process | an [`IApplicationProxy`] |
//! | | 10 | `OpenApplicationProxyWithFlags` (Switch 2) | `u32` flags at offset 0; then as command 0, with the `u64` placeholder at offset 8, not read | an [`IApplicationProxy`] |
//! | [`IApplicationProxy`] | 0 | `GetCommonStateGetter` | nothing | an [`ICommonStateGetter`] |
//! | | 1 | `GetSelfController` | nothing | an [`ISelfController`] |
//! | | 2 | `GetWindowController` | nothing | an [`IWindowController`] |
//! | | 3 | `GetAudioController` | nothing | an [`IAudioController`] |
//! | | 4 | `GetDisplayController` | nothing | an [`IDisplayController`] |
//! | | 11 | `GetLibraryAppletCreator` | nothing | an [`ILibraryAppletCreator`] |
//! | | 20 | `GetApplicationFunctions` | nothing | an [`IApplicationFunctions`] |
//! | | 40 | `GetPerformanceFunctions` (Switch 2) | nothing | an [`IPerformanceFunctions`] |
//! | | 60 | `GetGraphicsScalingParametersFunctions` (Switch 2) | nothing | an [`IGraphicsScalingParametersFunctions`] |
//! | | 1000 | `GetDebugFunctions` | nothing | an [`IDebugFunctions`] |
//! | [`ICommonStateGetter`] | 0 | `GetEventHandle` | nothing | copy handle: the applet's message event |
//! | | 1 | `ReceiveMessage` | nothing | `u32` [`AppletMessage`]; or [`NO_MESSAGES`] and nothing |
//! | | 5 | `GetOperationMode` | nothing | `u8` [`OperationMode`] |
//! | | 6 | `GetPerformanceMode` | nothing | `s32` [`PerformanceMode`] |
//! | | 9 | `GetCurrentFocusState` | nothing | `u8` [`FocusState`] |
//! | | 66 | `SetCpuBoostMode` | `u32` [`CpuBoostMode`], any word | nothing |
//! | | 900 | `SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled` | nothing | nothing |
//! | [`ISelfController`] | 0 | `Exit` | nothing | nothing; the process ends after the reply |
//! | | 10 | `SetScreenShotPermission` | `u32` [`ScreenshotPermission`] | nothing |
//! | | 11 | `SetOperationModeChangedNotification` | `bool` (a byte) | nothing |
//! | | 12 | `SetPerformanceModeChangedNotification` | `bool` | nothing |
//! | | 13 | `SetFocusHandlingMode` | `bool notify, bool background, bool suspend`, a byte each | nothing |
//! | | 14 | `SetRestartMessageEnabled` | `bool` | nothing |
//! | | 16 | `SetOutOfFocusSuspendingEnabled` | `bool` | nothing |
//! | | 91 | `GetAccumulatedSuspendedTickChangedEvent` | nothing | copy handle: the applet's suspended-tick event |
//! | | 2000 | `GetGraphicsSupervisor` (Switch 2) | nothing | copy handle: a new session on the graphics server's root |
//! | | 2400 | `SetRtcScreenSharingAudioEnabled` (Switch 2) | `bool` | nothing |
//! | [`IWindowController`] | 1 | `GetAppletResourceUserId` | nothing | `u64` the applet resource user id |
//! | | 10 | `AcquireForegroundRights` | nothing | nothing |
//! | [`IApplicationFunctions`] | 21 | `GetDesiredLanguage` | nothing | `u64` language code: [`DESIRED_LANGUAGE_CODE`] |
//! | | 40 | `NotifyRunning` | nothing | `bool` (a byte): true |
//! | | 65 | `IsGamePlayRecordingSupported` | nothing | `bool` (a byte): [`GAME_PLAY_RECORDING_SUPPORTED`] |
//! | | 67 | `SetGamePlayRecordingState` | `u32` [`GamePlayRecordingState`] | nothing |
//! | | 130 | `GetGpuErrorDetectedSystemEvent` | nothing | copy handle: the applet's GPU-error event |
//! | | 210 | `GetUnknownEvent210` (Eden's name `Unknown210`) | nothing | copy handle: the applet's unknown event |
//! | | 320 | `GetScreenShotAndContinuousRecordingInitiallyEnabled` (Switch 2) | nothing | `bool screenshot, bool recording`, a byte each, from the applet ([`Applet::screenshot_and_continuous_recording_initially_enabled`]) |
//! | | 321 | `Unknown321` (Switch 2) | nothing | `bool` (a byte): [`UNKNOWN_321_ENABLED`] |
//! | | 330 | `Unknown330` | nothing | `u8`: [`UNKNOWN_330_VALUE`] |
//! | [`IGraphicsScalingParametersFunctions`] (Switch 2) | 10 | `GetGraphicsScalingParameters` | nothing | a [`GraphicsScalingParameters`]: three `u32`s |
//! | | 1050 | `SetGraphicsScaleChangedNotificationEnabled` | `bool` (a byte) | nothing |
//!
//! [`IAudioController`], [`IDisplayController`], [`ILibraryAppletCreator`] and
//! [`IDebugFunctions`] have no command here: libnx opens them at start-up and sends them nothing.
//! [`IPerformanceFunctions`] has none either: PR #4588 names its two commands and implements
//! neither (see "The Switch 2 additions").
//!
//! Layouts, each the client's and the reference's:
//!
//! - `OpenApplicationProxy`: libnx `_appletGetSessionProxy` (`serviceDispatchIn(.., cmd_id,
//!   reserved, .in_send_pid = true, .in_num_handles = 1, .in_handles = { prochandle },
//!   .out_num_objects = 1)`, with `CUR_PROCESS_HANDLE`); Eden `OpenApplicationProxy(Out<..>,
//!   ClientProcessId pid, InCopyHandle<KProcess>)`, whose id is the kernel's
//!   (`eden/src/core/hle/service/cmif_serialization.h`, `InProcessId`: `pid = ctx.GetPID()`).
//! - The `Get...` commands that return an interface: libnx `_appletCmdGetSession` (no payload,
//!   one out object); Eden `Out<SharedPointer<..>>`.
//! - `GetEventHandle`, `GetAccumulatedSuspendedTickChangedEvent`,
//!   `GetGpuErrorDetectedSystemEvent`: libnx `_appletCmdGetEvent` (one copied handle); Eden
//!   `OutCopyHandle<KReadableEvent>`. `GetUnknownEvent210`: no libnx client; Eden
//!   `OutCopyHandle<KReadableEvent>`.
//! - `GetDesiredLanguage`: libnx `appletGetDesiredLanguage` (`_appletCmdNoInOutU64`); Eden
//!   `Out<u64>`.
//! - `SetCpuBoostMode`: libnx `appletSetCpuBoostMode` (`_appletCmdInU32NoOut`); Eden hands the
//!   request to `APM_Sys::SetCpuBoostMode`, which pops one `u32` enum at offset 0
//!   (`rp.PopEnum<CpuBoostMode>()`).
//! - `SetRestartMessageEnabled`: libnx `_appletCmdInBoolNoOut`; Eden a `bool` argument, as the
//!   other one-flag `Set...` commands below. `SetRtcScreenSharingAudioEnabled`: no libnx client;
//!   PR #4588 a `bool` argument ("SDK 23.0 serializes this command as one bool input with no
//!   output payload").
//! - `SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled`: libnx `_appletCmdNoIO`; Eden
//!   `Result SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled()`.
//! - `SetScreenShotPermission`, `SetGamePlayRecordingState`: libnx `_appletCmdInU32NoOut` (one
//!   `u32`; for 67 libnx sends `state != 0`, so 0 or 1); Eden one `ScreenshotPermission` or
//!   `GamePlayRecordingState` argument, each an `enum class : u32` (`am_types.h`), at offset 0.
//! - `IsGamePlayRecordingSupported`: libnx `_appletCmdNoInOutBool`; Eden `Out<bool>`.
//! - `ReceiveMessage`: libnx `_appletCmdNoInOutU32`; Eden `Out<AppletMessage>`, a `u32` enum
//!   (`am_types.h`), at offset 0.
//! - `Unknown330`: no libnx client; Eden `Out<u8>`.
//! - `GetOperationMode`, `GetCurrentFocusState`: libnx `_appletCmdNoInOutU8`; Eden
//!   `Out<OperationMode>` and `Out<FocusState>`, both `u8` enums (`am_types.h`).
//!   `GetPerformanceMode`: libnx `_appletCmdNoInOutU32`; Eden `Out<APM::PerformanceMode>`, an
//!   `s32` enum.
//! - `GetAppletResourceUserId`: libnx `_appletCmdNoInOutU64`; Eden `Out<AppletResourceUserId>`,
//!   a `ProcessId` (`eden/src/core/hle/service/cmif_types.h`), one `u64`.
//! - `Exit`: libnx `_appletSelfExit` (`_appletCmdNoIO`: no payload, nothing out); Eden
//!   `Result Exit()`.
//! - `NotifyRunning`: libnx `_appletCmdNoInOutBool` (a `u8`, of which it keeps bit 0); Eden
//!   `Out<bool>`, set to true.
//! - The `Set...` commands with one flag: libnx `_appletCmdInBoolNoOut` (one `u8`, 0 or 1); Eden a
//!   `bool` argument. `SetFocusHandlingMode`: libnx `_appletSetFocusHandlingMode`
//!   (`struct { u8 inval0; u8 inval1; u8 inval2; }`); Eden `(bool notify, bool background, bool
//!   suspend)`, at offsets 0, 1 and 2 (`cmif_serialization.h`, `ReadInArgument`: each argument at
//!   its own alignment).
//!
//! - `OpenApplicationProxyWithFlags`: no libnx client. PR #4588's
//!   `OpenApplicationProxyWithFlags(Out<..>, u32 flags, ClientProcessId pid,
//!   InCopyHandle<KProcess>)`: by `ReadInArgument`, `flags` at offset 0 and the process id's
//!   `u64` slot at offset 8 (its own alignment), 16 bytes in all. The slot is not read, as for
//!   command 0: the id is the kernel's.
//! - `GetScreenShotAndContinuousRecordingInitiallyEnabled`: PR #4588 `(Out<bool>, Out<bool>)`,
//!   at offsets 0 and 1. `Unknown321`: `Out<bool>`. `GetGraphicsScalingParameters`:
//!   `Out<GraphicsScalingParameters>`, a 12-byte struct of three `u32`s
//!   (`graphics_scaling_parameters_functions.h`: `static_assert(sizeof(GraphicsScalingParameters)
//!   == 12)`). `SetGraphicsScaleChangedNotificationEnabled`: a `bool` argument.
//!
//! A flag is read as true when its byte is not zero. Eden copies the byte into a C++ `bool`
//! (`std::memcpy` in `ReadInArgument`), which is defined only for 0 and 1; what it makes of
//! another byte is up to the compiler. Reading every other byte as true is a guess; a console
//! trace with such a byte would settle it. libnx sends only 0 and 1.
//!
//! # The applet
//!
//! Eden keeps one `Applet` per process the applet manager launched, and every interface of
//! `appletOE` reads and changes it. Here a run has one process, its application, and the
//! applet's state is the server's family state ([`Applet`]), which every session of the server
//! shares. What the console reports comes from the run's [`AppletSettings`], with Eden's
//! defaults ([`AppletSettings::eden_default`]):
//!
//! - **Operation mode:** docked, as Eden's `use_docked_mode` setting is by default on desktop
//!   (`eden/src/common/settings.h`: `ConsoleMode::Docked` outside Android).
//! - **Performance mode:** follows the operation mode, by Eden's rule
//!   ([`PerformanceMode::for_docked`]); it is not a setting of its own.
//! - **Focus state:** in focus. Eden's applet manager gives an application it launches the focus
//!   (`applet_manager.cpp`: `SetFocusState(FocusState::InFocus)`, "Applet was started by
//!   frontend, so it is foreground"), and nothing in a run changes it.
//!
//! A restore checks the graphics server's number against the run (see "The graphics server")
//! and reads every queued message through [`AppletMessage::from_raw`] (see "The messages"); it
//! has no kernel to check the applet's event ids against, so it believes them, and the commands
//! check them when they use them:
//!
//! - **The suspended-tick, GPU-error and unknown events** are checked each time
//!   `GetAccumulatedSuspendedTickChangedEvent`, `GetGpuErrorDetectedSystemEvent` or
//!   `GetUnknownEvent210` returns a kept one, before its handle is copied: the kernel's `clear_event` answers `InvalidHandle`
//!   for an id that is not an event this server made (another kind of object, another server's
//!   or the guest's event, or nothing), and the command answers that error, copies nothing and
//!   leaves the applet as it was (`a_restored_event_the_kernel_did_not_make_is_refused`). On
//!   the server's own event the clear changes nothing, as neither event is ever signalled
//!   (`a_restored_event_the_kernel_made_is_copied_unchanged`).
//! - **Not checked here: the message event** when `GetEventHandle` copies it. The kernel calls
//!   that check an event id (`clear_event`, `signal_event`) change its state, and the guest may
//!   have cleared the event itself, so no call checks it without changing what the guest sees;
//!   a restored message-event id that names another object is copied as it is. The lifecycle
//!   commands that signal or clear it answer the kernel's error for an id that is not an event
//!   (`a_failed_signal_still_keeps_the_lifecycle_change`,
//!   `a_failed_clear_still_keeps_the_lifecycle_change`, and for `ReceiveMessage`
//!   `a_failed_clear_leaves_the_message_to_be_received_again`).
//! - **Not checked here: whether the message event's state agrees with the restored
//!   messages.** The event's signalled state is the kernel's, saved with the kernel; a restore
//!   here cannot read it.
//!
//! # The message event
//!
//! `GetEventHandle` returns the applet's message event (Eden: the lifecycle manager's system
//! event), which is signalled while the applet has a message to receive. Of Eden's
//! `ShouldSignalSystemEvent`, this port keeps the terms of the messages it ports (see "The
//! messages", [`Lifecycle::message_available`]): for an application, focus-state-change
//! notifications are enabled and the focus state has changed since the applet last received
//! it; or an operation-mode or performance-mode change is pending; or an unordered message is
//! queued. Of these only the first is ever raised in a run: the settings never change while it
//! runs, and nothing here pushes an unordered message; the others hold only in an applet a
//! snapshot restored. Eden's other terms (resume, exit, sleep, display requests, SD card
//! removal, temperature, battery, auto power-down, album events, and the library-applet focus
//! terms) are left out: nothing here raises them. A port that raises one adds its term, and its
//! place in [`Lifecycle::pop_message_in_order_of_priority`]. So the event is signalled from
//! launch (Eden's `SetFocusState(InFocus)` at launch finds a focus change pending, with
//! notifications on); `ReceiveMessage` takes that message and clears the event, and so does
//! libnx's `SetFocusHandlingMode(0, 0, 1)`, which turns the notifications off.
//!
//! Eden signals or clears the event in `SignalSystemEventIfNeeded`
//! (`eden/src/core/hle/service/am/lifecycle_manager.cpp` @ cf654052: `m_system_event.Signal`
//! when a message became available, `m_system_event.Clear` when one stopped being available),
//! when the cached "a message is available" differs from `ShouldSignalSystemEvent`. Here the
//! event's state is compared before and after each command that changes an input of that
//! condition ([`Lifecycle`]): the same transitions, because in this port only those commands
//! change the inputs, and Eden calls `SignalSystemEventIfNeeded` after each of them. A message
//! that stops being available clears the event ([`Context::clear_event`]): libnx's
//! `SetFocusHandlingMode(0, 0, 1)` at start-up does (Eden: `SetFocusStateChangedNotificationEnabled`
//! in `lifecycle_manager.h`), so libnx's `appletMainLoop`, which waits on the event, does not
//! wake on every poll.
//!
//! The events are created the first time a command returns them, not when the applet is: the
//! kernel interface is reached only while a request is handled. A message event created then is
//! signalled exactly when the condition above holds, which is the state Eden's event is in at
//! that moment. The suspended-tick, GPU-error and unknown events are never signalled, as in
//! Eden (only the applet's constructor and the commands that hand them out name them,
//! `applet.cpp`).
//!
//! # The messages
//!
//! `ReceiveMessage` (Eden: `ICommonStateGetter::ReceiveMessage`, `common_state_getter.cpp` @
//! cf654052) takes the applet's next message and answers its word, or [`NO_MESSAGES`] when it
//! has none. Which message is next is Eden's `LifecycleManager::PopMessageInOrderOfPriority`
//! (`lifecycle_manager.cpp` @ cf654052) for an application, over the terms this port keeps, in
//! Eden's order ([`Lifecycle::pop_message_in_order_of_priority`]): the focus change (while its
//! notifications are on), the operation-mode change, the performance-mode change, then the
//! unordered messages, oldest first. As Eden's `PopMessage` does after the pop, the message
//! event is cleared if no message is left; here the clear comes just before the pop (see "Where
//! this differs from Eden"), and only when a message was available, as Eden clears only when
//! its cached "a message is available" was true
//! (`with_no_message_pending_nothing_clears_the_message_event`). Each `ReceiveMessage` reads at
//! most the first two queued messages, however long the queue a snapshot restored
//! (`per_call_work_does_not_grow_with_the_queue`). `lifecycle_manager.cpp`, `.h`,
//! `am_results.h`, the `AppletMessage` values and `ReceiveMessage` are the same in Eden-NX2 @
//! 50fa86bb and Eden PR #4588 @ e5fb3b97, and the scoping doc has no row for them: the Switch 2
//! changes nothing here.
//!
//! So a freshly launched application's first `ReceiveMessage` answers
//! [`AppletMessage::FocusStateChanged`] (Eden's `m_has_focus_state_changed{true}`), and the next
//! answers [`NO_MESSAGES`], with the event clear
//! (`receive_message_answers_the_focus_change_then_no_messages`). A title that waits on the
//! event and then receives therefore stops waking once it has the focus change.
//!
//! The message state is saved with the applet. A queued message is saved as its word and read
//! back through [`AppletMessage::from_raw`], so a snapshot whose queue holds a word Eden does not
//! declare, or `None` (0, which no message carries), is refused with
//! [`SnapshotError::Decode`]`(postcard::Error::SerdeDeCustom)`, and the server is left as it
//! was (`a_snapshot_with_an_invalid_queued_message_is_refused`); `postcard` keeps no text for a
//! refusal of this kind. A restore accepts any declared word in the queue, including the
//! messages Eden raises only through its flags (`FocusStateChanged`, `OperationModeChanged`,
//! `PerformanceModeChanged`) or sends only to other applets: such a message is delivered as
//! Eden's `PopMessage` would deliver it had it been pushed
//! (`a_restored_queue_is_delivered_in_order_after_the_flags`).
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `am/tests.rs`:
//!
//! - **A failed clear in `ReceiveMessage` leaves the message pending.** Eden's event cannot fail
//!   to clear. Here the kernel refuses to clear a restored message-event id that is not an
//!   event; `ReceiveMessage`, which clears the event before it takes the last message, then
//!   answers the kernel's error and takes nothing, so the message is not lost undelivered and
//!   the next `ReceiveMessage` takes it again
//!   (`a_failed_clear_leaves_the_message_to_be_received_again`). The `ISelfController` commands
//!   instead keep their change on a failed clear ("The applet").
//!
//! - **`Exit` replies before the process ends.** Eden's `Exit` calls `KProcess::Terminate` on
//!   the applet's process inside the handler and then answers success. Here the method asks
//!   the kernel to end the caller's process ([`Context::end_process`]), and the kernel does so
//!   once the success reply is in the caller's message buffer and the request is recorded, as
//!   the services spec says (approved 2026-11-16). libnx reads that reply: on success it sleeps
//!   until it is ended (`libnx/nx/source/services/applet.c`, `_appletExitProcess`). The order is
//!   pinned on the real kernel by `exit_replies_and_then_the_kernel_ends_the_process`
//!   (`tests/am_on_the_kernel.rs`), whose caller is a thread the scheduler runs, so a reply
//!   written after the end would not reach it; and in the kernel by
//!   `a_handler_that_ends_the_process_replies_first_and_then_releases_everything`. The fake has
//!   no process to end, so `exit_answers_success_and_asks_once_to_end_the_process` pins only
//!   that `Exit` answers success and asks the kernel once, not the order.
//!   Eden ends the applet's process, which is the caller's here (the next item).
//! - **Any process is the application.** Eden finds the applet by the caller's process id and
//!   answers `ResultUnknown` when no applet was launched for it
//!   (`IApplicationProxyService::GetAppletFromProcessId`). starquake has no applet manager, so
//!   every process id opens the application's proxy, and the applet resource user id is the
//!   process id the proxy was opened with, which is what Eden's is for its applet
//!   (`Applet::Applet`: `aruid.pid = process->GetProcessId()`)
//!   (`any_process_id_opens_the_proxy_and_is_its_applet_resource_user_id`).
//! - **No process id, or no process handle, is refused.** Eden takes the id from the kernel
//!   whether or not the request asked for it, and reads copy handle 0 without checking that there
//!   is one. Here a request without the id is answered [`UNKNOWN`] (Eden's answer for a process
//!   with no applet), and one without a copied handle [`INVALID_CMIF_REQUEST`]
//!   (`open_application_proxy_needs_the_process_id_and_the_process_handle`). libnx sends both.
//!   The handle is not otherwise used: Eden keeps it for the display layer manager, which is not
//!   ported.
//!
//! # The Switch 2 additions
//!
//! The scoping doc's `am` row (`docs/switch2-eden-diff-scoping.md`, "Services (HLE)") names
//! them: `appletOE` command 10 as the new `OpenApplicationProxy`, `IApplicationProxy` 60
//! (graphics scale) and 40, and `IApplicationFunctions` 320 and 321. Each is ported as PR #4588
//! has it (`ISelfController` 2000 is the next section):
//!
//! - **`OpenApplicationProxyWithFlags` (10)** checks its flags and is then
//!   `OpenApplicationProxy`, through the same method, so the proxy it returns is the one command
//!   0 returns. PR's check is `R_UNLESS(flags == 0 || flags == 2, ResultUnknown)`, with the
//!   comment that 0 is what most titles send and 2 what one title sends, and that no other value
//!   has been observed: [`PROXY_FLAGS_ACCEPTED`], answered [`UNKNOWN`] otherwise. What the flags
//!   mean is not known; they are not kept. The checks run in PR's order: the flags first, then
//!   command 0's (the process id, then the process handle).
//! - **`GetPerformanceFunctions` (40)** returns an [`IPerformanceFunctions`]. PR #4588's table
//!   for it is `{0, nullptr, "IsCpuOverclockEnabled"}, {1, nullptr, "SetCpuOverclockEnabled"}`:
//!   no command is implemented, so here both go to the unknown-command policy.
//! - **`GetGraphicsScalingParametersFunctions` (60)** returns an
//!   [`IGraphicsScalingParametersFunctions`], of whose commands PR implements two:
//!   `GetGraphicsScalingParameters` (10) answers [`GraphicsScalingParameters::PR_4588`], and
//!   `SetGraphicsScaleChangedNotificationEnabled` (1050) keeps its flag in the applet
//!   ([`Applet::graphics_scale_changed_notification_enabled`], initially false: `applet.h`,
//!   `bool graphics_scale_changed_notification_enabled{}`). PR reads the flag nowhere else, so
//!   it changes no answer. PR's other rows (1000, 1010, 1020, 1030 and 1040, all `...ForDebug`)
//!   are `nullptr`, and go to the unknown-command policy.
//! - **`IApplicationFunctions` 320** (`GetScreenShotAndContinuousRecordingInitiallyEnabled`)
//!   answers PR's expressions over the applet (see "Screenshots and recording").
//! - **`IApplicationFunctions` 321** (`Unknown321`) answers false ([`UNKNOWN_321_ENABLED`]).
//!   PR names it `Unknown321`, and its comment says what the SDK uses the bool for; both the
//!   meaning and the value are PR's guesses.
//!
//! Where these differ from PR #4588:
//!
//! - **A payload too short for a read field is refused** with [`INVALID_CMIF_REQUEST`] and
//!   changes nothing: `OpenApplicationProxyWithFlags` with fewer than 4 bytes (it reads only the
//!   flags), `SetGraphicsScaleChangedNotificationEnabled` with none. PR's `ReadInArgument` copies
//!   the bytes at the argument's offset whatever the message's size. What counts as the payload
//!   depends on the session ([`Request::payload`]): on a domain it is the domain header's `size`
//!   less the `SFCI` header, so a request that carries fewer bytes is refused; on an ordinary
//!   CMIF session it runs to the end of the message's raw data, padding included, so a request
//!   with fewer argument bytes is refused only when the raw data also ends before the field, and
//!   otherwise reads the padding and succeeds. Only the domain case is pinned:
//!   `open_application_proxy_with_flags_checks_its_flags_first` (0 bytes, then 1, 2 and 3 bytes
//!   with libnx's exact domain `size`, refused, and 4 bytes, accepted),
//!   `set_graphics_scale_changed_notification_enabled_keeps_its_flag` (0 bytes) and, for
//!   `SetScreenShotPermission` and `SetGamePlayRecordingState` (Eden's handlers, under the same
//!   `ReadInArgument`), `a_payload_too_short_for_the_word_is_refused_and_changes_nothing` (0, 1,
//!   2 and 3 bytes with libnx's exact domain `size`, refused, and 4 bytes, accepted).
//! - The process id and handle checks are command 0's ("Where this differs from Eden").
//!
//! # Screenshots and recording
//!
//! 320 answers, in this order, PR's `screenshot_permission != ScreenshotPermission::Disable`
//! ([`ScreenshotPermission::screenshots_enabled`]) and `game_play_recording_supported &&
//! game_play_recording_state == GamePlayRecordingState::Enabled`
//! ([`GamePlayRecordingState::continuous_recording_enabled`]), over the applet
//! (`eden-pr4588/src/core/hle/service/am/service/application_functions.cpp` @ e5fb3b97). The
//! fields follow the commands that set them, as in PR:
//!
//! - **`ISelfController::SetScreenShotPermission` (10)** keeps the `u32` it is sent in the
//!   applet ([`Applet::screenshot_permission`]), initially `Inherit` (`applet.h`,
//!   `screenshot_permission{}`). Any word is kept, as PR's `enum class : u32` keeps it; only
//!   `Disable` (2) turns screenshots off, so a word that differs from 2 above its low byte
//!   leaves them on (`set_screen_shot_permission_keeps_its_word_and_320_follows_it`).
//! - **`IApplicationFunctions::SetGamePlayRecordingState` (67)** keeps its `u32` likewise
//!   ([`Applet::game_play_recording_state`]), initially `Disabled` (`applet.h`).
//! - **`IApplicationFunctions::IsGamePlayRecordingSupported` (65)** answers
//!   `game_play_recording_supported`, which no command of Eden, Eden-NX2 or PR writes, so it
//!   keeps `applet.h`'s `false`: [`GAME_PLAY_RECORDING_SUPPORTED`], a constant here. So 320's
//!   second byte is false whatever the state
//!   (`set_game_play_recording_state_keeps_its_word_and_recording_stays_off`); the expression
//!   itself is pinned for both values of the support flag by
//!   `continuous_recording_needs_support_and_the_enabled_state`. A port of something that
//!   writes the flag makes it applet state.
//!
//! `IApplicationFunctions::InitializeGamePlayRecording` (66), which libnx sends before it sets
//! the state on, is not ported: it goes to the unknown-command policy.
//!
//! # The title's start-up commands
//!
//! A Switch 2 title sends two commands at start-up that the scoping doc has no row for:
//! `ICommonStateGetter` 900 and `IApplicationFunctions` 130. Each is ported as Eden, Eden-NX2
//! and PR #4588 have it, the same handler in all three:
//!
//! - **`SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled` (900)** sets the applet's
//!   `request_exit_to_library_applet_at_execute_next_program_enabled` to true
//!   ([`Applet::request_exit_to_library_applet_at_execute_next_program_enabled`], initially
//!   false: `applet.h`) and answers nothing. No reference reads the flag, so it changes no
//!   answer.
//! - **`GetGpuErrorDetectedSystemEvent` (130)** copies a handle to the applet's GPU-error event
//!   ([`Applet::gpu_error_detected_event`]), made the first time, as the suspended-tick event
//!   is, and never signalled: no reference signals `gpu_error_detected_event`.
//!
//! `IApplicationFunctions` 330 (`Unknown330`, marked 20.0.0+) sits next to 320 and 321 in PR's
//! table and is the same in all three references, `Out<u8>` set to 0
//! ([`UNKNOWN_330_VALUE`]); the scoping doc has no row for it either.
//!
//! # The title's later start-up commands
//!
//! Once its parental-control check passes, a Switch 2 title sends five more commands of
//! `appletOE` at start-up. Each is ported as Eden has it; the handlers of the four that Eden has
//! are the same in Eden-NX2 and PR #4588:
//!
//! - **`GetDesiredLanguage` (21)** answers [`DESIRED_LANGUAGE_CODE`], `en-US`. Eden reads the
//!   title's supported languages from its NACP when it can, and asks `ns` for the first entry of
//!   the system language's priority list that the title supports. The NACP is not available to
//!   this service, so this is Eden's answer when the metadata is missing (supported languages 0,
//!   which accepts any language), under Eden's default system language (see the constant). Two
//!   divergences follow, both pinned by
//!   `get_desired_language_answers_eden_s_answer_with_no_metadata`: a title whose NACP leaves
//!   out American English is answered `en-US` all the same, where Eden answers the first
//!   language of the priority list the title supports; and the system language is not a
//!   setting here. Wiring the NACP through is a proposal (`docs/proposals.md`).
//! - **`GetUnknownEvent210` (210)** copies a handle to the applet's unknown event
//!   ([`Applet::unknown_event`], Eden's `unknown_event`), made the first time, as the GPU-error
//!   event is, and kept by the applet: every call copies a handle to the same event. Nothing
//!   signals it: no reference signals `unknown_event` (Eden's library-applet accessor hands the
//!   same event out, and signals it nowhere either).
//! - **`SetCpuBoostMode` (66)** answers success for any word, as Eden does. Eden forwards it to
//!   `apm:sys`, whose controller checks the word (`u32(mode) < BOOST_MODE_TO_CONFIG_MAP.size()`,
//!   3: [`CpuBoostMode::from_raw`]) and, for a mode, records performance configuration 7, 13 or
//!   15 as the boost mode's and calls `SetClockSpeed`, which does nothing (a `TODO`); a word
//!   that is not a mode is only logged. **apm is not affected here**: starquake's `apm` keeps no
//!   performance configuration, and no ported command reads one (`apm`'s `ISession`
//!   `GetPerformanceConfiguration` and `apm:sys` are not ported, [`crate::apm`]), so the check
//!   changes only what is logged, and Eden's recorded configuration is not kept. Eden's
//!   `GetPerformanceMode` depends on the operation mode alone, not on the configuration, so
//!   `ICommonStateGetter::GetPerformanceMode` answers the same after it
//!   (`set_cpu_boost_mode_answers_success_and_keeps_nothing`). A port of `apm`'s configuration
//!   commands makes the boost configuration state that `appletOE` and `apm` share.
//! - **`SetRestartMessageEnabled` (14)** keeps its flag in the applet
//!   ([`Applet::resume_notification_enabled`], Eden's `m_resume_notification_enabled`,
//!   initially false), and
//!   touches nothing else: Eden's `SetResumeNotificationEnabled` sets the flag without calling
//!   `SignalSystemEventIfNeeded`
//!   (`set_restart_message_enabled_keeps_its_flag_and_leaves_the_message_event`). Eden reads the
//!   flag only in `RequestResumeNotification`, which raises the `Resume` message when an applet
//!   is resumed; nothing here resumes the applet, so the flag changes no answer yet.
//! - **`SetRtcScreenSharingAudioEnabled` (2400)** is a Switch 2 command, a fact of the PR #4588
//!   fork only (`eden-pr4588/src/core/hle/service/am/service/self_controller.cpp` @ e5fb3b97,
//!   line 84; Eden and Eden-NX2 have no row 2400). PR only logs its flag and answers success,
//!   and so does this port; nothing is kept
//!   (`set_rtc_screen_sharing_audio_enabled_answers_success_and_keeps_nothing`).
//!
//! The scoping doc has no row for any of the five. A payload too short for the argument of 66,
//! 14 or 2400 is refused with [`INVALID_CMIF_REQUEST`] and changes nothing, where Eden reads
//! past the argument whatever the message's size: the same divergence, with the same session
//! rule, as "The Switch 2 additions" lists for 10 and 1050. It is pinned for the domain case by
//! `the_later_commands_refuse_a_payload_too_short_and_change_nothing` (0 to 3 bytes for 66, and
//! none for 14 and 2400, with libnx's exact domain `size`, refused; the whole argument accepted).
//!
//! # The graphics server
//!
//! `ISelfController` 2000 returns a session to the graphics server's root, which the title then
//! speaks TIPC on (`docs/specs/graphics-server.md`, "The root"; `eden-nx2/NX2_NOTES.md` row 17,
//! observed). `am` may not depend on the graphics server's crate, so it is given the number of
//! the root's server ([`Applet::with_graphics_server`]): whoever puts the services together
//! gives it the same number the service registry's private entry for the root holds
//! (`crate::sm`, "Private entries"). It hands out a session on it with
//! [`Reply::copy_service`], and the root's server opens its root interface for the session.
//!
//! - **A copied handle.** Both references copy it: Eden PR #4588
//!   (`eden-pr4588/src/core/hle/service/am/service/self_controller.cpp` @ e5fb3b97,
//!   `GetGraphicsSupervisor`: `IPC::ResponseBuilder rb{ctx, 2, 1}`, `PushCopyObjects`) and
//!   Eden-NX2 (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `ReturnRawSession`:
//!   `PushCopyObjects`). The reply is the result and the one copied handle.
//! - **A new session each time: conflict.** PR keeps the session it made the first time and
//!   copies a handle to that same session on every later call; NX2 makes a new session on every
//!   call. NX2's is followed, as the one a title ran on (row 17), by the graphics-server spec's
//!   house rule (decision 8). The root's state is its server's, which every session shares.
//!   Against PR, then, the two differ only in how many sessions exist. Against NX2 they differ
//!   in the state too: each NX2 session owns its own objects and syncpoint events, so a second
//!   one starts fresh, where a second session here sees the first one's bootstrap phase and
//!   204 events (`starquake_gsv::root`, "Where this differs from the references"). A trace of a
//!   title calling 2000 twice would settle it. Pinned by
//!   `get_graphics_supervisor_makes_a_new_session_on_every_call`.
//! - **A run without a graphics server** answers [`UNKNOWN`], starquake's own choice: no
//!   reference has an `am` without one
//!   (`get_graphics_supervisor_without_a_graphics_server_is_unknown`).
//! - **The number is the run's wiring, not the guest's state.** It is saved with the applet,
//!   but a restore keeps the number the server was built with: a snapshot whose number differs
//!   (another server, or one where the run has none, or none where it has one) is refused with
//!   [`SnapshotError::FamilyWiring`], and the server is left as it was
//!   (`a_snapshot_with_another_graphics_server_is_refused`). Otherwise a snapshot could point
//!   2000 at any server.
//!
//! [`INVALID_CMIF_REQUEST`]: crate::framework::result::INVALID_CMIF_REQUEST
//! [`Reply::copy_service`]: crate::framework::Reply::copy_service
//! [`SnapshotError::FamilyWiring`]: crate::framework::SnapshotError::FamilyWiring
//! [`SnapshotError::Decode`]: crate::framework::SnapshotError::Decode

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;
use starquake_trace::ServiceId;

use std::collections::VecDeque;

use crate::apm::PerformanceMode;
use crate::commands;
use crate::framework::result::code;
use crate::framework::{
    Context, Fault, Interfaces, Outcome, Reader, Reply, Request, Server, UnknownPolicy, Writer,
};

#[cfg(test)]
mod tests;

/// The name `appletOE` is registered under (`eden/src/core/hle/service/am/am.cpp`,
/// `LoopProcess`: `RegisterNamedService("appletOE", ..)`).
pub const OE_NAME: &str = "appletOE";

/// Eden's `ResultUnknown` (`eden/src/core/hle/result.h`: `constexpr Result
/// ResultUnknown(UINT32_MAX)`), which `OpenApplicationProxy` answers when it finds no applet for
/// the caller.
pub const UNKNOWN: ResultCode = ResultCode::from_raw(u32::MAX);

/// `ErrorModule::AM` (`eden/src/core/hle/result.h` @ cf654052: `AM = 128`).
pub const AM_MODULE: u32 = 128;

/// `ResultNoMessages` (`eden/src/core/hle/service/am/am_results.h` @ cf654052: `constexpr Result
/// ResultNoMessages{ErrorModule::AM, 3}`; the same in Eden-NX2 @ 50fa86bb and Eden PR #4588 @
/// e5fb3b97), which `ReceiveMessage` answers when the applet has no message (module docs, "The
/// messages"). libnx tells it apart by the same word: `appletMainLoop` skips a failure whose
/// `R_VALUE(rc) == 0x680`, and `appletGetMessage` checks `MAKERESULT(128, 3)`
/// (`libnx/nx/source/services/applet.c`).
pub const NO_MESSAGES: ResultCode = code(AM_MODULE, 3);

/// The flags `OpenApplicationProxyWithFlags` accepts (module docs, "The Switch 2 additions"):
/// Eden PR #4588 (`eden-pr4588/src/core/hle/service/am/service/application_proxy_service.cpp` @
/// e5fb3b97, `OpenApplicationProxyWithFlags`): `R_UNLESS(flags == 0 || flags == 2,
/// ResultUnknown)`. Any other value is answered [`UNKNOWN`].
pub const PROXY_FLAGS_ACCEPTED: [u32; 2] = [0, 2];

/// Whether the applet supports game-play recording, as `IsGamePlayRecordingSupported` answers
/// and `GetScreenShotAndContinuousRecordingInitiallyEnabled` reads (module docs, "Screenshots
/// and recording"): `applet.h`'s `bool game_play_recording_supported{}`, false, in Eden @
/// cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97 alike. No command of any of the
/// three writes the field, so it is a constant here rather than applet state.
pub const GAME_PLAY_RECORDING_SUPPORTED: bool = false;

/// What `IApplicationFunctions` 330 answers: `Unknown330(Out<u8> out)` sets `*out = 0` in Eden
/// (`eden/src/core/hle/service/am/service/application_functions.cpp` @ cf654052), Eden-NX2
/// (`eden-nx2/src/core/hle/service/am/service/application_functions.cpp` @ 50fa86bb) and Eden
/// PR #4588 (`eden-pr4588/src/core/hle/service/am/service/application_functions.cpp` @
/// e5fb3b97) alike. The scoping doc has no row for it. A guess of the references': the
/// command's meaning is not known, and a console trace of the reply would confirm the value.
pub const UNKNOWN_330_VALUE: u8 = 0;

/// An applet's screenshot permission, as `SetScreenShotPermission` sets it (Eden PR #4588,
/// `eden-pr4588/src/core/hle/service/am/am_types.h` @ e5fb3b97: `enum class
/// ScreenshotPermission : u32 { Inherit = 0, Enable = 1, Disable = 2 }`; libnx's
/// `AppletScreenShotPermission` has the same values).
///
/// It holds any `u32`: the reference copies the request's word into the enum whatever its
/// value (`cmif_serialization.h`, `ReadInArgument`), and only compares it with `Disable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreenshotPermission(u32);

impl ScreenshotPermission {
    /// `Inherit = 0`: the applet's initial permission (`applet.h`, `ScreenshotPermission
    /// screenshot_permission{}`).
    pub const INHERIT: Self = Self(0);
    /// `Enable = 1`.
    pub const ENABLE: Self = Self(1);
    /// `Disable = 2`.
    pub const DISABLE: Self = Self(2);

    /// The permission a request's word names.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// The permission's word.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Whether screenshots are enabled under it: PR #4588's `screenshot_permission !=
    /// ScreenshotPermission::Disable` (`application_functions.cpp`,
    /// `GetScreenShotAndContinuousRecordingInitiallyEnabled`).
    #[must_use]
    pub const fn screenshots_enabled(self) -> bool {
        self.0 != Self::DISABLE.0
    }
}

/// An applet's game-play recording state, as `SetGamePlayRecordingState` sets it (Eden PR
/// #4588, `eden-pr4588/src/core/hle/service/am/am_types.h` @ e5fb3b97: `enum class
/// GamePlayRecordingState : u32 { Disabled, Enabled }`).
///
/// It holds any `u32`, as [`ScreenshotPermission`] does: the reference stores the request's
/// word whatever its value, and only compares it with `Enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GamePlayRecordingState(u32);

impl GamePlayRecordingState {
    /// `Disabled = 0`: the applet's initial state (`applet.h`, `GamePlayRecordingState
    /// game_play_recording_state{GamePlayRecordingState::Disabled}`).
    pub const DISABLED: Self = Self(0);
    /// `Enabled = 1`.
    pub const ENABLED: Self = Self(1);

    /// The state a request's word names.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// The state's word.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Whether continuous recording is enabled in this state, for an applet that does or does
    /// not support recording: PR #4588's `game_play_recording_supported &&
    /// game_play_recording_state == GamePlayRecordingState::Enabled` (`application_functions.cpp`,
    /// `GetScreenShotAndContinuousRecordingInitiallyEnabled`).
    #[must_use]
    pub const fn continuous_recording_enabled(self, supported: bool) -> bool {
        supported && self.0 == Self::ENABLED.0
    }
}

/// `LanguageCode::EN_US` (`eden/src/core/hle/service/set/settings_types.h` @ cf654052: `EN_US =
/// 0x00000053552D6E65`; the same in Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97): the bytes
/// of `en-US`, zero-padded to eight, read as a little-endian `u64`.
pub const LANGUAGE_CODE_EN_US: u64 = 0x0000_0053_552D_6E65;

/// What `IApplicationFunctions::GetDesiredLanguage` answers: Eden's answer for an application
/// whose metadata it cannot read, under Eden's default system language (module docs, "The
/// title's later start-up commands").
///
/// Eden starts from `supported_languages = 0` ("Default to 0 (all languages supported)") and
/// reads the NACP's only when `GetApplicationMetadata` finds it
/// (`eden/src/core/hle/service/am/service/application_functions.cpp` @ cf654052,
/// `GetDesiredLanguage`). `IReadOnlyApplicationControlDataInterface::GetApplicationDesiredLanguage`
/// (`eden/src/core/hle/service/ns/read_only_application_control_data_interface.cpp`) then takes
/// the system language (`Settings::values.language_index`, `Language::EnglishAmerican` by
/// default: `eden/src/common/settings.h`; index 1 of `available_language_codes`, `EN_US`:
/// `set/settings_types.h`), turns it into `ApplicationLanguage::AmericanEnglish`
/// (`ns/language.cpp`, `ConvertToApplicationLanguage`), and returns the first language of its
/// priority list (`priority_list_american_english`, whose first entry is `AmericanEnglish`) for
/// which `supported_languages == 0 || (supported_languages & supported_flag) == supported_flag`
/// holds, so with 0 the first. `ConvertApplicationLanguageToLanguageCode` turns that back into
/// `EN_US` (`ConvertToLanguageCode`). The same files are identical in Eden-NX2 @ 50fa86bb and
/// Eden PR #4588 @ e5fb3b97.
pub const DESIRED_LANGUAGE_CODE: u64 = LANGUAGE_CODE_EN_US;

/// A CPU boost mode, as `ICommonStateGetter::SetCpuBoostMode` sends it (`nn::oe::CpuBoostMode`;
/// `eden/src/core/hle/service/apm/apm_controller.h` @ cf654052: `enum class CpuBoostMode : u32
/// { Normal = 0, FastLoad = 1, Partial = 2 }`; libnx's `ApmCpuBoostMode` has the same values,
/// naming the third `Type2`).
///
/// Every value Eden accepts is a variant: `Controller::SetFromCpuBoostMode` (`apm_controller.cpp`)
/// checks `u32(mode) < BOOST_MODE_TO_CONFIG_MAP.size()`, a map of three entries, so exactly the
/// words 0, 1 and 2 are modes ([`CpuBoostMode::from_raw`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBoostMode {
    /// `Normal = 0`: boost off.
    Normal,
    /// `FastLoad = 1`.
    FastLoad,
    /// `Partial = 2`.
    Partial,
}

impl CpuBoostMode {
    /// Every mode, in Eden's declaration order (which is also the order of their words).
    pub const ALL: [Self; 3] = [Self::Normal, Self::FastLoad, Self::Partial];

    /// The mode's word: the one place each value is written down.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::FastLoad => 1,
            Self::Partial => 2,
        }
    }

    /// The mode whose word is `raw`, or `None` for a word Eden's check refuses (module docs,
    /// "The title's later start-up commands").
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.raw() == raw)
    }
}

/// What `IApplicationFunctions` 321 answers: PR #4588's `Unknown321` sets `*out_enabled = false`
/// (`eden-pr4588/src/core/hle/service/am/service/application_functions.cpp` @ e5fb3b97). A guess
/// of PR's: the command's meaning is not known, and a console trace of the reply would confirm
/// the value.
pub const UNKNOWN_321_ENABLED: bool = false;

/// The graphics scaling parameters `GetGraphicsScalingParameters` answers (Eden PR #4588,
/// `eden-pr4588/src/core/hle/service/am/service/graphics_scaling_parameters_functions.h` @
/// e5fb3b97, `struct GraphicsScalingParameters`, 12 bytes). The reply carries the fields in
/// declaration order, each a little-endian `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsScalingParameters {
    /// `graphics_scale`.
    pub graphics_scale: u32,
    /// `screen_ratio_numerator`.
    pub screen_ratio_numerator: u32,
    /// `screen_ratio_denominator`.
    pub screen_ratio_denominator: u32,
}

impl GraphicsScalingParameters {
    /// PR #4588's answer, `*out_parameters = {10, 1, 1}`, with its comment: "Prototype policy:
    /// lowest SDK scale (10 maps to public scale 0), unit screen ratio. The ratio field
    /// interpretation is inferred from the adjacent debug API." All three values, and what the
    /// ratio fields mean, are PR's guesses; a console trace of the reply would confirm them.
    pub const PR_4588: Self = Self {
        graphics_scale: 10,
        screen_ratio_numerator: 1,
        screen_ratio_denominator: 1,
    };

    /// Writes the parameters as the reply carries them.
    fn write(&self, payload: &mut Writer) {
        payload
            .write_u32(self.graphics_scale)
            .write_u32(self.screen_ratio_numerator)
            .write_u32(self.screen_ratio_denominator);
    }
}

/// What the console reports as its operation mode (`nn::oe::OperationMode`;
/// `eden/src/core/hle/service/am/am_types.h`, `enum class OperationMode : u8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationMode {
    /// `Handheld = 0`.
    Handheld,
    /// `Docked = 1`.
    Docked,
}

impl OperationMode {
    /// The mode as the byte a reply carries.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Handheld => 0,
            Self::Docked => 1,
        }
    }

    /// Whether the console is docked.
    #[must_use]
    pub const fn is_docked(self) -> bool {
        matches!(self, Self::Docked)
    }
}

/// An applet's focus state (`eden/src/core/hle/service/am/am_types.h`, `enum class FocusState :
/// u8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusState {
    /// `InFocus = 1`.
    InFocus,
    /// `NotInFocus = 2`.
    NotInFocus,
    /// `Background = 3`.
    Background,
}

impl FocusState {
    /// The state as the byte a reply carries.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::InFocus => 1,
            Self::NotInFocus => 2,
            Self::Background => 3,
        }
    }
}

/// How an applet is suspended when it loses the focus
/// (`eden/src/core/hle/service/am/lifecycle_manager.h`, `enum class FocusHandlingMode : u32`).
/// Nothing in a run takes the focus away, so the mode changes no answer yet; it is kept because
/// two commands set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusHandlingMode {
    /// `AlwaysSuspend = 0`.
    AlwaysSuspend,
    /// `SuspendHomeSleep = 1`: the lifecycle manager's initial mode
    /// (`m_focus_handling_mode{FocusHandlingMode::SuspendHomeSleep}`).
    SuspendHomeSleep,
    /// `NoSuspend = 2`.
    NoSuspend,
}

/// A message an applet receives with `ReceiveMessage` (`nn::am::AppletMessage`;
/// `eden/src/core/hle/service/am/am_types.h` @ cf654052, `enum class AppletMessage : u32`, the
/// same values in Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97).
///
/// Every value Eden declares is a variant, except `None = 0`, which Eden's
/// `PopMessageInOrderOfPriority` returns for "no message" and `ReceiveMessage` never sends: here
/// "no message" is `Option::None` ([`Lifecycle::pop_message_in_order_of_priority`]). So the type
/// holds exactly the values a message can carry, and a snapshot that names any other word is
/// refused (it is read through [`AppletMessage::from_raw`]; module docs, "The messages").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub enum AppletMessage {
    /// `ChangeIntoForeground = 1`.
    ChangeIntoForeground,
    /// `ChangeIntoBackground = 2`.
    ChangeIntoBackground,
    /// `Exit = 4`.
    Exit,
    /// `ApplicationExited = 6`.
    ApplicationExited,
    /// `FocusStateChanged = 15`.
    FocusStateChanged,
    /// `Resume = 16`.
    Resume,
    /// `DetectShortPressingHomeButton = 20`.
    DetectShortPressingHomeButton,
    /// `DetectLongPressingHomeButton = 21`.
    DetectLongPressingHomeButton,
    /// `DetectShortPressingPowerButton = 22`.
    DetectShortPressingPowerButton,
    /// `DetectMiddlePressingPowerButton = 23`.
    DetectMiddlePressingPowerButton,
    /// `DetectLongPressingPowerButton = 24`.
    DetectLongPressingPowerButton,
    /// `RequestToPrepareSleep = 25`.
    RequestToPrepareSleep,
    /// `FinishedSleepSequence = 26`.
    FinishedSleepSequence,
    /// `SleepRequiredByHighTemperature = 27`.
    SleepRequiredByHighTemperature,
    /// `SleepRequiredByLowBattery = 28`.
    SleepRequiredByLowBattery,
    /// `AutoPowerDown = 29`.
    AutoPowerDown,
    /// `OperationModeChanged = 30`.
    OperationModeChanged,
    /// `PerformanceModeChanged = 31`.
    PerformanceModeChanged,
    /// `DetectReceivingCecSystemStandby = 32`.
    DetectReceivingCecSystemStandby,
    /// `SdCardRemoved = 33`.
    SdCardRemoved,
    /// `LaunchApplicationRequested = 50`.
    LaunchApplicationRequested,
    /// `RequestToDisplay = 51`.
    RequestToDisplay,
    /// `ShowApplicationLogo = 55`.
    ShowApplicationLogo,
    /// `HideApplicationLogo = 56`.
    HideApplicationLogo,
    /// `ForceHideApplicationLogo = 57`.
    ForceHideApplicationLogo,
    /// `FloatingApplicationDetected = 60`.
    FloatingApplicationDetected,
    /// `DetectShortPressingCaptureButton = 90`.
    DetectShortPressingCaptureButton,
    /// `DetectLongPressingCaptureButton = 91`.
    DetectLongPressingCaptureButton,
    /// `AlbumScreenShotTaken = 92`.
    AlbumScreenShotTaken,
    /// `AlbumRecordingSaved = 93`.
    AlbumRecordingSaved,
    /// `StartupLogoDisappeared = 95`.
    StartupLogoDisappeared,
}

impl AppletMessage {
    /// Every message, in Eden's declaration order (which is also the order of their words).
    pub const ALL: [Self; 31] = [
        Self::ChangeIntoForeground,
        Self::ChangeIntoBackground,
        Self::Exit,
        Self::ApplicationExited,
        Self::FocusStateChanged,
        Self::Resume,
        Self::DetectShortPressingHomeButton,
        Self::DetectLongPressingHomeButton,
        Self::DetectShortPressingPowerButton,
        Self::DetectMiddlePressingPowerButton,
        Self::DetectLongPressingPowerButton,
        Self::RequestToPrepareSleep,
        Self::FinishedSleepSequence,
        Self::SleepRequiredByHighTemperature,
        Self::SleepRequiredByLowBattery,
        Self::AutoPowerDown,
        Self::OperationModeChanged,
        Self::PerformanceModeChanged,
        Self::DetectReceivingCecSystemStandby,
        Self::SdCardRemoved,
        Self::LaunchApplicationRequested,
        Self::RequestToDisplay,
        Self::ShowApplicationLogo,
        Self::HideApplicationLogo,
        Self::ForceHideApplicationLogo,
        Self::FloatingApplicationDetected,
        Self::DetectShortPressingCaptureButton,
        Self::DetectLongPressingCaptureButton,
        Self::AlbumScreenShotTaken,
        Self::AlbumRecordingSaved,
        Self::StartupLogoDisappeared,
    ];

    /// The message's word, as `ReceiveMessage`'s reply carries it: the one place each value is
    /// written down.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::ChangeIntoForeground => 1,
            Self::ChangeIntoBackground => 2,
            Self::Exit => 4,
            Self::ApplicationExited => 6,
            Self::FocusStateChanged => 15,
            Self::Resume => 16,
            Self::DetectShortPressingHomeButton => 20,
            Self::DetectLongPressingHomeButton => 21,
            Self::DetectShortPressingPowerButton => 22,
            Self::DetectMiddlePressingPowerButton => 23,
            Self::DetectLongPressingPowerButton => 24,
            Self::RequestToPrepareSleep => 25,
            Self::FinishedSleepSequence => 26,
            Self::SleepRequiredByHighTemperature => 27,
            Self::SleepRequiredByLowBattery => 28,
            Self::AutoPowerDown => 29,
            Self::OperationModeChanged => 30,
            Self::PerformanceModeChanged => 31,
            Self::DetectReceivingCecSystemStandby => 32,
            Self::SdCardRemoved => 33,
            Self::LaunchApplicationRequested => 50,
            Self::RequestToDisplay => 51,
            Self::ShowApplicationLogo => 55,
            Self::HideApplicationLogo => 56,
            Self::ForceHideApplicationLogo => 57,
            Self::FloatingApplicationDetected => 60,
            Self::DetectShortPressingCaptureButton => 90,
            Self::DetectLongPressingCaptureButton => 91,
            Self::AlbumScreenShotTaken => 92,
            Self::AlbumRecordingSaved => 93,
            Self::StartupLogoDisappeared => 95,
        }
    }

    /// The message whose word is `raw`, or [`NotAnAppletMessage`] for a word that is not one of
    /// [`AppletMessage::ALL`]'s (`0`, Eden's `None`, among them).
    pub fn from_raw(raw: u32) -> Result<Self, NotAnAppletMessage> {
        Self::ALL
            .into_iter()
            .find(|message| message.raw() == raw)
            .ok_or(NotAnAppletMessage(raw))
    }
}

impl TryFrom<u32> for AppletMessage {
    type Error = NotAnAppletMessage;

    fn try_from(raw: u32) -> Result<Self, Self::Error> {
        Self::from_raw(raw)
    }
}

impl From<AppletMessage> for u32 {
    fn from(message: AppletMessage) -> Self {
        message.raw()
    }
}

/// A word that is not an [`AppletMessage`]: what [`AppletMessage::from_raw`] answers for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{0} is not an applet message")]
pub struct NotAnAppletMessage(pub u32);

/// What the console reports to the application: the run's configuration of `appletOE` (see the
/// module docs, "The applet").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppletSettings {
    /// The operation mode; the performance mode follows it ([`PerformanceMode::for_docked`]).
    pub operation_mode: OperationMode,
    /// The application's focus state.
    pub focus_state: FocusState,
}

impl AppletSettings {
    /// Eden's: docked, and in focus (module docs, "The applet").
    #[must_use]
    pub const fn eden_default() -> Self {
        Self {
            operation_mode: OperationMode::Docked,
            focus_state: FocusState::InFocus,
        }
    }

    /// The performance mode the console reports.
    #[must_use]
    pub const fn performance_mode(&self) -> PerformanceMode {
        PerformanceMode::for_docked(self.operation_mode.is_docked())
    }
}

/// Which changes the applet asked to be told of: three of Eden's `LifecycleManager` flags, each
/// initially true (`lifecycle_manager.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Notifications {
    /// `m_focus_state_changed_notification_enabled`.
    pub focus_state_changed: bool,
    /// `m_operation_mode_changed_notification_enabled`.
    pub operation_mode_changed: bool,
    /// `m_performance_mode_changed_notification_enabled`.
    pub performance_mode_changed: bool,
}

impl Default for Notifications {
    fn default() -> Self {
        Self {
            focus_state_changed: true,
            operation_mode_changed: true,
            performance_mode_changed: true,
        }
    }
}

/// The part of Eden's `LifecycleManager` that the ported commands read and change, with Eden's
/// initial values (`lifecycle_manager.h`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lifecycle {
    /// The notifications the applet asked for.
    pub notifications: Notifications,
    /// `m_has_focus_state_changed`, initially true; `ReceiveMessage` clears it when it delivers
    /// [`AppletMessage::FocusStateChanged`].
    pub has_focus_state_changed: bool,
    /// `m_focus_handling_mode`.
    pub focus_handling_mode: FocusHandlingMode,
    /// `m_has_operation_mode_changed`, initially false; `ReceiveMessage` clears it when it
    /// delivers [`AppletMessage::OperationModeChanged`]. Nothing in a run sets it (module docs,
    /// "The messages").
    pub has_operation_mode_changed: bool,
    /// `m_has_performance_mode_changed`, initially false; `ReceiveMessage` clears it when it
    /// delivers [`AppletMessage::PerformanceModeChanged`]. Nothing in a run sets it.
    pub has_performance_mode_changed: bool,
    /// `m_unordered_messages`, initially empty: messages delivered first in, first out, after
    /// every flag above. Nothing in a run pushes one (module docs, "The messages").
    pub unordered_messages: VecDeque<AppletMessage>,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            notifications: Notifications::default(),
            has_focus_state_changed: true,
            focus_handling_mode: FocusHandlingMode::SuspendHomeSleep,
            has_operation_mode_changed: false,
            has_performance_mode_changed: false,
            unordered_messages: VecDeque::new(),
        }
    }
}

impl Lifecycle {
    /// Whether the applet has a message to receive, so its message event is signalled: the terms
    /// of `LifecycleManager::ShouldSignalSystemEvent` for an application that this port keeps
    /// (module docs, "The message event"). It holds exactly when
    /// [`Lifecycle::pop_message_in_order_of_priority`] would return a message.
    #[must_use]
    pub fn message_available(&self) -> bool {
        (self.notifications.focus_state_changed && self.has_focus_state_changed)
            || !self.unordered_messages.is_empty()
            || self.has_operation_mode_changed
            || self.has_performance_mode_changed
    }

    /// Whether a message would still be available after
    /// [`Lifecycle::pop_message_in_order_of_priority`]: what `ReceiveMessage` decides on before
    /// it takes anything (module docs, "The messages").
    ///
    /// It takes the message from a copy of the flags and of at most the first two queued
    /// messages, not of the whole queue: a pop removes at most one queued message, and
    /// [`Lifecycle::message_available`] reads only whether the queue is empty, so the first two
    /// decide the answer. Its cost does not grow with the queue
    /// (`per_call_work_does_not_grow_with_the_queue`), and it agrees with a pop on a full copy
    /// (`message_available_after_pop_agrees_with_a_pop`).
    fn message_available_after_pop(&self) -> bool {
        let mut next = Self {
            notifications: self.notifications,
            has_focus_state_changed: self.has_focus_state_changed,
            focus_handling_mode: self.focus_handling_mode,
            has_operation_mode_changed: self.has_operation_mode_changed,
            has_performance_mode_changed: self.has_performance_mode_changed,
            unordered_messages: self.unordered_messages.iter().take(2).copied().collect(),
        };
        next.pop_message_in_order_of_priority();
        next.message_available()
    }

    /// `LifecycleManager::PopMessageInOrderOfPriority` for an application (Eden's
    /// `m_is_application`), over the terms this port keeps: the first of these that holds is
    /// taken and returned, or `None` when none does (Eden's `AppletMessage::None`):
    ///
    /// 1. a focus change, while focus-change notifications are on: clears
    ///    [`Lifecycle::has_focus_state_changed`], [`AppletMessage::FocusStateChanged`];
    /// 2. an operation-mode change: clears [`Lifecycle::has_operation_mode_changed`],
    ///    [`AppletMessage::OperationModeChanged`];
    /// 3. a performance-mode change: clears [`Lifecycle::has_performance_mode_changed`],
    ///    [`AppletMessage::PerformanceModeChanged`];
    /// 4. the oldest of [`Lifecycle::unordered_messages`], removed.
    ///
    /// A focus change pending while the notifications are off stays pending, as in Eden.
    pub fn pop_message_in_order_of_priority(&mut self) -> Option<AppletMessage> {
        if self.notifications.focus_state_changed && self.has_focus_state_changed {
            self.has_focus_state_changed = false;
            return Some(AppletMessage::FocusStateChanged);
        }
        if self.has_operation_mode_changed {
            self.has_operation_mode_changed = false;
            return Some(AppletMessage::OperationModeChanged);
        }
        if self.has_performance_mode_changed {
            self.has_performance_mode_changed = false;
            return Some(AppletMessage::PerformanceModeChanged);
        }
        self.unordered_messages.pop_front()
    }

    /// `LifecycleManager::SetFocusHandlingMode(suspend)`.
    pub fn set_focus_handling_mode(&mut self, suspend: bool) {
        self.focus_handling_mode = match (self.focus_handling_mode, suspend) {
            // "Disallow suspension."
            (FocusHandlingMode::AlwaysSuspend | FocusHandlingMode::SuspendHomeSleep, false) => {
                FocusHandlingMode::NoSuspend
            }
            // "Allow suspension temporally."
            (FocusHandlingMode::NoSuspend, true) => FocusHandlingMode::SuspendHomeSleep,
            (unchanged, _) => unchanged,
        };
    }

    /// `LifecycleManager::SetOutOfFocusSuspendingEnabled(enabled)`.
    pub fn set_out_of_focus_suspending_enabled(&mut self, enabled: bool) {
        self.focus_handling_mode = match (self.focus_handling_mode, enabled) {
            // "Allow suspension temporally."
            (FocusHandlingMode::AlwaysSuspend, false) => FocusHandlingMode::SuspendHomeSleep,
            // "Allow suspension."
            (FocusHandlingMode::SuspendHomeSleep | FocusHandlingMode::NoSuspend, true) => {
                FocusHandlingMode::AlwaysSuspend
            }
            (unchanged, _) => unchanged,
        };
    }
}

/// The application's applet: the `appletOE` server's family state (see the module docs, "The
/// applet").
///
/// A command changes it in place, never on a copy: a copy would cost as much as the applet's
/// message queue, which a restored snapshot may make long
/// (`per_call_work_does_not_grow_with_the_queue`). A command that changes it and calls the
/// kernel keeps its change before it calls, so a kernel object it created is never forgotten
/// and an error the kernel reports leaves the change kept. The one exception is
/// `ReceiveMessage`, which clears the message event before it takes the message, and takes
/// nothing if the clear fails (module docs, "The messages").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Applet {
    settings: AppletSettings,
    lifecycle: Lifecycle,
    /// The message event, once a command has returned it.
    message_event: Option<ObjectId>,
    /// The suspended-tick event, once a command has returned it.
    suspended_tick_event: Option<ObjectId>,
    /// The server of the graphics server's root, which `GetGraphicsSupervisor` hands sessions
    /// on out (module docs, "The graphics server"); `None` in a run without one.
    graphics_server: Option<ServiceId>,
    /// PR #4588's `graphics_scale_changed_notification_enabled`, which
    /// `SetGraphicsScaleChangedNotificationEnabled` sets (module docs, "The Switch 2 additions").
    graphics_scale_changed_notification_enabled: bool,
    /// `screenshot_permission`, which `SetScreenShotPermission` sets (module docs, "Screenshots
    /// and recording").
    screenshot_permission: ScreenshotPermission,
    /// `game_play_recording_state`, which `SetGamePlayRecordingState` sets (module docs,
    /// "Screenshots and recording").
    game_play_recording_state: GamePlayRecordingState,
    /// `request_exit_to_library_applet_at_execute_next_program_enabled`, which
    /// `SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled` sets (module docs, "The
    /// title's start-up commands").
    request_exit_to_library_applet_at_execute_next_program_enabled: bool,
    /// The GPU-error event, once a command has returned it (module docs, "The title's start-up
    /// commands").
    gpu_error_detected_event: Option<ObjectId>,
    /// Eden's `unknown_event`, which `GetUnknownEvent210` returns, once a command has returned
    /// it (module docs, "The title's later start-up commands").
    unknown_event: Option<ObjectId>,
    /// The lifecycle manager's `m_resume_notification_enabled`, which `SetRestartMessageEnabled`
    /// sets (module docs, "The title's later start-up commands"). Kept here rather than in
    /// [`Lifecycle`], which holds the state that decides whether a message is available: this
    /// flag is not a term of it.
    resume_notification_enabled: bool,
}

impl Applet {
    /// A freshly launched application's applet, reporting `settings`, in a run without a
    /// graphics server.
    #[must_use]
    pub fn new(settings: AppletSettings) -> Self {
        Self {
            settings,
            lifecycle: Lifecycle::default(),
            message_event: None,
            suspended_tick_event: None,
            graphics_server: None,
            graphics_scale_changed_notification_enabled: false,
            screenshot_permission: ScreenshotPermission::INHERIT,
            game_play_recording_state: GamePlayRecordingState::DISABLED,
            request_exit_to_library_applet_at_execute_next_program_enabled: false,
            gpu_error_detected_event: None,
            unknown_event: None,
            resume_notification_enabled: false,
        }
    }

    /// The same applet, in a run whose graphics server's root is the server known as `root`
    /// (the number the service registry has for it: `Registry::private_service`).
    #[must_use]
    pub fn with_graphics_server(mut self, root: ServiceId) -> Self {
        self.graphics_server = Some(root);
        self
    }

    /// The server `GetGraphicsSupervisor` hands sessions on out, if the run has one.
    #[must_use]
    pub fn graphics_server(&self) -> Option<ServiceId> {
        self.graphics_server
    }

    /// Whether the applet asked to be told when the graphics scale changes
    /// (`SetGraphicsScaleChangedNotificationEnabled`); initially false.
    #[must_use]
    pub fn graphics_scale_changed_notification_enabled(&self) -> bool {
        self.graphics_scale_changed_notification_enabled
    }

    /// What the applet reports.
    #[must_use]
    pub fn settings(&self) -> &AppletSettings {
        &self.settings
    }

    /// The applet's lifecycle state.
    #[must_use]
    pub fn lifecycle(&self) -> &Lifecycle {
        &self.lifecycle
    }

    /// The message event, if a command has returned it yet.
    #[must_use]
    pub fn message_event(&self) -> Option<ObjectId> {
        self.message_event
    }

    /// The suspended-tick event, if a command has returned it yet.
    #[must_use]
    pub fn suspended_tick_event(&self) -> Option<ObjectId> {
        self.suspended_tick_event
    }

    /// The GPU-error event (`GetGpuErrorDetectedSystemEvent`), if a command has returned it
    /// yet.
    #[must_use]
    pub fn gpu_error_detected_event(&self) -> Option<ObjectId> {
        self.gpu_error_detected_event
    }

    /// The unknown event (`GetUnknownEvent210`), if a command has returned it yet.
    #[must_use]
    pub fn unknown_event(&self) -> Option<ObjectId> {
        self.unknown_event
    }

    /// Whether the applet asked to be told when it is resumed (`SetRestartMessageEnabled`);
    /// initially false, as Eden's `bool m_resume_notification_enabled{}` (`lifecycle_manager.h`).
    #[must_use]
    pub fn resume_notification_enabled(&self) -> bool {
        self.resume_notification_enabled
    }

    /// The screenshot permission (`SetScreenShotPermission`); initially
    /// [`ScreenshotPermission::INHERIT`].
    #[must_use]
    pub fn screenshot_permission(&self) -> ScreenshotPermission {
        self.screenshot_permission
    }

    /// The game-play recording state (`SetGamePlayRecordingState`); initially
    /// [`GamePlayRecordingState::DISABLED`].
    #[must_use]
    pub fn game_play_recording_state(&self) -> GamePlayRecordingState {
        self.game_play_recording_state
    }

    /// Whether the application asked to exit to the library applet when it executes its next
    /// program (`SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled`); initially false.
    #[must_use]
    pub fn request_exit_to_library_applet_at_execute_next_program_enabled(&self) -> bool {
        self.request_exit_to_library_applet_at_execute_next_program_enabled
    }

    /// What `GetScreenShotAndContinuousRecordingInitiallyEnabled` answers: whether screenshots
    /// are enabled, then whether continuous recording is (module docs, "Screenshots and
    /// recording").
    #[must_use]
    pub fn screenshot_and_continuous_recording_initially_enabled(&self) -> (bool, bool) {
        (
            self.screenshot_permission.screenshots_enabled(),
            self.game_play_recording_state
                .continuous_recording_enabled(GAME_PLAY_RECORDING_SUPPORTED),
        )
    }

    /// Changes the server's applet's lifecycle state with `change`, in place, then signals the
    /// message event if a message became available, or clears it if one stopped being
    /// available: Eden's `SignalSystemEventIfNeeded` after the change (module docs, "The message
    /// event"). Before the event exists there is nothing to signal or clear: it is made in the
    /// state the condition says. The change is kept when the signal or clear fails.
    fn update(cx: &mut Context<'_>, change: impl FnOnce(&mut Lifecycle)) -> Result<(), Fault> {
        let applet = cx.family::<Self>()?;
        let before = applet.lifecycle.message_available();
        change(&mut applet.lifecycle);
        let after = applet.lifecycle.message_available();
        let message_event = applet.message_event;
        if after
            && !before
            && let Some(event) = message_event
        {
            cx.signal_event(event)?;
        }
        if before
            && !after
            && let Some(event) = message_event
        {
            cx.clear_event(event)?;
        }
        Ok(())
    }

    /// Takes the server's applet's next message
    /// ([`Lifecycle::pop_message_in_order_of_priority`]), clearing the message event first if
    /// no message will be left: Eden's `LifecycleManager::PopMessage`, which calls
    /// `SignalSystemEventIfNeeded` after the pop. Taking a message never makes one available,
    /// so the event is only ever cleared here, never signalled; with no message available
    /// before, nothing is cleared, as Eden clears only when its cached "a message is available"
    /// was true.
    ///
    /// The clear is decided on the applet as it is ([`Lifecycle::message_available_after_pop`])
    /// and done before the pop, so a clear that fails (a restored event id that is not an
    /// event) reports the kernel's error and takes nothing (module docs, "Where this differs
    /// from Eden").
    fn pop_message(cx: &mut Context<'_>) -> Result<Option<AppletMessage>, Fault> {
        let applet = cx.family::<Self>()?;
        let before = applet.lifecycle.message_available();
        let after = applet.lifecycle.message_available_after_pop();
        let message_event = applet.message_event;
        if before
            && !after
            && let Some(event) = message_event
        {
            cx.clear_event(event)?;
        }
        Ok(cx
            .family::<Self>()?
            .lifecycle
            .pop_message_in_order_of_priority())
    }

    /// The server's applet's message event, created (and signalled if a message is available)
    /// the first time.
    fn message_event_or_create(cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        if let Some(event) = cx.family::<Self>()?.message_event {
            return Ok(event);
        }
        let event = cx.create_event()?;
        // Kept before it is signalled, so the event the server holds a reference to is never
        // lost; signalling an event just created fails only if the kernel lost it.
        let applet = cx.family::<Self>()?;
        applet.message_event = Some(event);
        if applet.lifecycle.message_available() {
            cx.signal_event(event)?;
        }
        Ok(event)
    }

    /// The server's applet's suspended-tick event, created the first time. It is never
    /// signalled.
    fn suspended_tick_event_or_create(cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        unsignalled_event_or_create(cx, |applet| &mut applet.suspended_tick_event)
    }

    /// The server's applet's GPU-error event, created the first time. It is never signalled.
    fn gpu_error_detected_event_or_create(cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        unsignalled_event_or_create(cx, |applet| &mut applet.gpu_error_detected_event)
    }

    /// The server's applet's unknown event (Eden's `unknown_event`), created the first time.
    /// It is never signalled.
    fn unknown_event_or_create(cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        unsignalled_event_or_create(cx, |applet| &mut applet.unknown_event)
    }
}

/// The event the server's applet keeps in `slot`, or a new one the kernel makes, kept there: how
/// the applet makes each of its events that nothing signals.
///
/// An event already in `slot` is checked first with [`Context::clear_event`], which the kernel
/// answers `InvalidHandle` for any id that is not an event this server made (the kernel's
/// `IpcContext::clear_event`), so a restored id that names another object, or nothing, is
/// refused before a handle to it is copied (module docs, "The applet"). On an event the server
/// made the clear changes nothing: nothing signals these events, and the guest is given only
/// their readable end, which it cannot signal.
fn unsignalled_event_or_create(
    cx: &mut Context<'_>,
    slot: fn(&mut Applet) -> &mut Option<ObjectId>,
) -> Result<ObjectId, Fault> {
    if let Some(event) = *slot(cx.family::<Applet>()?) {
        cx.clear_event(event)?;
        return Ok(event);
    }
    let event = cx.create_event()?;
    *slot(cx.family::<Applet>()?) = Some(event);
    Ok(event)
}

/// Copies the event `get` returns (one of the applet's, which `get` keeps in the applet before
/// any error it reports, so an event it made is never forgotten) into the reply.
fn reply_with_applet_event(
    cx: &mut Context<'_>,
    reply: &mut Reply,
    get: impl FnOnce(&mut Context<'_>) -> Result<ObjectId, Fault>,
) -> Outcome {
    reply.copy_handle(get(cx)?);
    reply.ok()
}

/// Reads a flag: one byte, true when it is not zero (see the module docs, "Layouts").
fn read_flag(reader: &mut Reader<'_>) -> Result<bool, Fault> {
    Ok(reader.read_u8()? != 0)
}

/// The root interface of an `appletOE` session (`IApplicationProxyService`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IApplicationProxyService;

/// The application's proxy (`IApplicationProxy`): the way to every other interface.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IApplicationProxy {
    /// The applet resource user id: the process id the proxy was opened with.
    aruid: u64,
}

/// `ICommonStateGetter`. The state it reads is the applet's.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ICommonStateGetter;

/// `ISelfController`. The state it changes is the applet's.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ISelfController;

/// `IWindowController`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IWindowController {
    /// The applet resource user id, from the proxy.
    aruid: u64,
}

/// `IApplicationFunctions`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IApplicationFunctions;

/// `IAudioController`: no command is ported (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IAudioController;

/// `IDisplayController`: no command is ported (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IDisplayController;

/// `ILibraryAppletCreator`: no command is ported (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ILibraryAppletCreator;

/// `IDebugFunctions`: no command is ported (module docs, "Commands").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IDebugFunctions;

/// `IPerformanceFunctions` (Switch 2): PR #4588 implements none of its commands (module docs,
/// "The Switch 2 additions").
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IPerformanceFunctions;

/// `IGraphicsScalingParametersFunctions` (Switch 2). The state it changes is the applet's.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IGraphicsScalingParametersFunctions;

/// The ids of the commands ported here, per interface (Eden's tables, each row cited).
pub mod command {
    /// `appletOE`'s commands (`IApplicationProxyService::IApplicationProxyService`).
    pub mod proxy_service {
        /// `{0, D<&IApplicationProxyService::OpenApplicationProxy>, "OpenApplicationProxy"}`.
        pub const OPEN_APPLICATION_PROXY: u32 = 0;
        /// Switch 2. Eden PR #4588
        /// (`eden-pr4588/src/core/hle/service/am/service/application_proxy_service.cpp` @
        /// e5fb3b97): `{10, D<&IApplicationProxyService::OpenApplicationProxyWithFlags>,
        /// "OpenApplicationProxyWithFlags"}`; the scoping doc's `am` row (`appletOE` command 10
        /// as the new `OpenApplicationProxy`).
        pub const OPEN_APPLICATION_PROXY_WITH_FLAGS: u32 = 10;
    }

    /// `IApplicationProxy`'s commands (`IApplicationProxy::IApplicationProxy`).
    pub mod proxy {
        /// `{0, D<&IApplicationProxy::GetCommonStateGetter>, "GetCommonStateGetter"}`.
        pub const GET_COMMON_STATE_GETTER: u32 = 0;
        /// `{1, D<&IApplicationProxy::GetSelfController>, "GetSelfController"}`.
        pub const GET_SELF_CONTROLLER: u32 = 1;
        /// `{2, D<&IApplicationProxy::GetWindowController>, "GetWindowController"}`.
        pub const GET_WINDOW_CONTROLLER: u32 = 2;
        /// `{3, D<&IApplicationProxy::GetAudioController>, "GetAudioController"}`.
        pub const GET_AUDIO_CONTROLLER: u32 = 3;
        /// `{4, D<&IApplicationProxy::GetDisplayController>, "GetDisplayController"}`.
        pub const GET_DISPLAY_CONTROLLER: u32 = 4;
        /// `{11, D<&IApplicationProxy::GetLibraryAppletCreator>, "GetLibraryAppletCreator"}`.
        pub const GET_LIBRARY_APPLET_CREATOR: u32 = 11;
        /// `{20, D<&IApplicationProxy::GetApplicationFunctions>, "GetApplicationFunctions"}`.
        pub const GET_APPLICATION_FUNCTIONS: u32 = 20;
        /// Switch 2. Eden PR #4588
        /// (`eden-pr4588/src/core/hle/service/am/service/application_proxy.cpp` @ e5fb3b97):
        /// `{40, D<&IApplicationProxy::GetPerformanceFunctions>, "GetPerformanceFunctions"}`.
        pub const GET_PERFORMANCE_FUNCTIONS: u32 = 40;
        /// Switch 2. Eden PR #4588 (`application_proxy.cpp` @ e5fb3b97): `{60,
        /// D<&IApplicationProxy::GetGraphicsScalingParametersFunctions>,
        /// "GetGraphicsScalingParametersFunctions"}`.
        pub const GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS: u32 = 60;
        /// `{1000, D<&IApplicationProxy::GetDebugFunctions>, "GetDebugFunctions"}`.
        pub const GET_DEBUG_FUNCTIONS: u32 = 1000;
    }

    /// `ICommonStateGetter`'s commands (`ICommonStateGetter::ICommonStateGetter`).
    pub mod common_state_getter {
        /// `{0, D<&ICommonStateGetter::GetEventHandle>, "GetEventHandle"}`.
        pub const GET_EVENT_HANDLE: u32 = 0;
        /// `{1, D<&ICommonStateGetter::ReceiveMessage>, "ReceiveMessage"}`, the same row in Eden
        /// @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/common_state_getter.cpp` in each); libnx
        /// `_appletReceiveMessage` sends 1 to `ICommonStateGetter` with `_appletCmdNoInOutU32`.
        pub const RECEIVE_MESSAGE: u32 = 1;
        /// `{5, D<&ICommonStateGetter::GetOperationMode>, "GetOperationMode"}`.
        pub const GET_OPERATION_MODE: u32 = 5;
        /// `{6, D<&ICommonStateGetter::GetPerformanceMode>, "GetPerformanceMode"}`.
        pub const GET_PERFORMANCE_MODE: u32 = 6;
        /// `{9, D<&ICommonStateGetter::GetCurrentFocusState>, "GetCurrentFocusState"}`.
        pub const GET_CURRENT_FOCUS_STATE: u32 = 9;
        /// `{900,
        /// D<&ICommonStateGetter::SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled>,
        /// "SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled"}, //11.0.0+`, the same row
        /// in Eden @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/common_state_getter.cpp` in each); libnx
        /// `appletSetRequestExitToLibraryAppletAtExecuteNextProgramEnabled` sends 900 to
        /// `ICommonStateGetter`. The scoping doc has no row for it.
        pub const SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED: u32 = 900;
        /// `{66, &ICommonStateGetter::SetCpuBoostMode, "SetCpuBoostMode"}`, the same row in
        /// Eden @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/common_state_getter.cpp` in each); libnx
        /// `appletSetCpuBoostMode` sends 66 to `ICommonStateGetter` with
        /// `_appletCmdInU32NoOut`. The scoping doc has no row for it.
        pub const SET_CPU_BOOST_MODE: u32 = 66;
    }

    /// `ISelfController`'s commands (`ISelfController::ISelfController`).
    pub mod self_controller {
        /// `{0, D<&ISelfController::Exit>, "Exit"}`.
        pub const EXIT: u32 = 0;
        /// `{10, D<&ISelfController::SetScreenShotPermission>, "SetScreenShotPermission"}`;
        /// libnx `appletSetScreenShotPermission` sends 10 with `_appletCmdInU32NoOut`.
        pub const SET_SCREEN_SHOT_PERMISSION: u32 = 10;
        /// `{11, D<&ISelfController::SetOperationModeChangedNotification>, ..}`.
        pub const SET_OPERATION_MODE_CHANGED_NOTIFICATION: u32 = 11;
        /// `{12, D<&ISelfController::SetPerformanceModeChangedNotification>, ..}`.
        pub const SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION: u32 = 12;
        /// `{13, D<&ISelfController::SetFocusHandlingMode>, "SetFocusHandlingMode"}`.
        pub const SET_FOCUS_HANDLING_MODE: u32 = 13;
        /// `{16, D<&ISelfController::SetOutOfFocusSuspendingEnabled>, ..}`.
        pub const SET_OUT_OF_FOCUS_SUSPENDING_ENABLED: u32 = 16;
        /// `{91, D<&ISelfController::GetAccumulatedSuspendedTickChangedEvent>, ..}`.
        pub const GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT: u32 = 91;
        /// The Switch 2 addition that returns a session to the graphics server's root (module
        /// docs, "The graphics server"). Eden PR #4588
        /// (`eden-pr4588/src/core/hle/service/am/service/self_controller.cpp` @ e5fb3b97):
        /// `{2000, &ISelfController::GetGraphicsSupervisor, "GetGraphicsSupervisor"}`; Eden-NX2
        /// (`eden-nx2/src/core/hle/service/am/service/self_controller.cpp` @ 50fa86bb):
        /// `{2000, &ISelfController::GetNx2GraphicsSystemSession, ..}`; `eden-nx2/NX2_NOTES.md`
        /// row 17 (observed).
        pub const GET_GRAPHICS_SUPERVISOR: u32 = 2000;
        /// `{14, D<&ISelfController::SetRestartMessageEnabled>, "SetRestartMessageEnabled"}`,
        /// the same row in Eden @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/self_controller.cpp` in each); libnx
        /// `appletSetRestartMessageEnabled` sends 14 to `ISelfController` with
        /// `_appletCmdInBoolNoOut`. The scoping doc has no row for it.
        pub const SET_RESTART_MESSAGE_ENABLED: u32 = 14;
        /// Switch 2, a fact of the PR #4588 fork only. Eden PR #4588
        /// (`eden-pr4588/src/core/hle/service/am/service/self_controller.cpp` @ e5fb3b97, line
        /// 84): `{2400, D<&ISelfController::SetRtcScreenSharingAudioEnabled>,
        /// "SetRtcScreenSharingAudioEnabled"}`. Eden and Eden-NX2 have no row 2400, and the
        /// scoping doc has no row for it.
        pub const SET_RTC_SCREEN_SHARING_AUDIO_ENABLED: u32 = 2400;
    }

    /// `IWindowController`'s commands (`IWindowController::IWindowController`).
    pub mod window_controller {
        /// `{1, D<&IWindowController::GetAppletResourceUserId>, "GetAppletResourceUserId"}`.
        pub const GET_APPLET_RESOURCE_USER_ID: u32 = 1;
        /// `{10, D<&IWindowController::AcquireForegroundRights>, "AcquireForegroundRights"}`.
        pub const ACQUIRE_FOREGROUND_RIGHTS: u32 = 10;
    }

    /// `IApplicationFunctions`' commands (`IApplicationFunctions::IApplicationFunctions`).
    pub mod application_functions {
        /// `{40, D<&IApplicationFunctions::NotifyRunning>, "NotifyRunning"}`.
        pub const NOTIFY_RUNNING: u32 = 40;
        /// `{65, D<&IApplicationFunctions::IsGamePlayRecordingSupported>,
        /// "IsGamePlayRecordingSupported"}`; libnx `appletIsGamePlayRecordingSupported` sends 65
        /// with `_appletCmdNoInOutBool`.
        pub const IS_GAME_PLAY_RECORDING_SUPPORTED: u32 = 65;
        /// `{67, D<&IApplicationFunctions::SetGamePlayRecordingState>,
        /// "SetGamePlayRecordingState"}`; libnx `appletSetGamePlayRecordingState` sends 67 with
        /// `_appletCmdInU32NoOut`.
        pub const SET_GAME_PLAY_RECORDING_STATE: u32 = 67;
        /// `{130, D<&IApplicationFunctions::GetGpuErrorDetectedSystemEvent>,
        /// "GetGpuErrorDetectedSystemEvent"}`, the same row in Eden @ cf654052, Eden-NX2 @
        /// 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/application_functions.cpp` in each); libnx
        /// `appletGetGpuErrorDetectedSystemEvent` sends 130 to `IApplicationFunctions` for an
        /// application (`_appletCmdGetEvent`). The scoping doc has no row for it.
        pub const GET_GPU_ERROR_DETECTED_SYSTEM_EVENT: u32 = 130;
        /// Switch 2. Eden PR #4588
        /// (`eden-pr4588/src/core/hle/service/am/service/application_functions.cpp` @
        /// e5fb3b97): `{320,
        /// D<&IApplicationFunctions::GetScreenShotAndContinuousRecordingInitiallyEnabled>,
        /// "GetScreenShotAndContinuousRecordingInitiallyEnabled"}`.
        pub const GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED: u32 = 320;
        /// Switch 2. Eden PR #4588 (`application_functions.cpp` @ e5fb3b97): `{321,
        /// D<&IApplicationFunctions::Unknown321>, "Unknown321"}`; PR's name.
        pub const UNKNOWN_321: u32 = 321;
        /// `{330, D<&IApplicationFunctions::Unknown330>, "Unknown330"}, // [20.0.0+]`, the same
        /// row in Eden @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/application_functions.cpp` in each); their name.
        /// The scoping doc has no row for it.
        pub const UNKNOWN_330: u32 = 330;
        /// `{21, D<&IApplicationFunctions::GetDesiredLanguage>, "GetDesiredLanguage"}`, the same
        /// row in Eden @ cf654052, Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97
        /// (`src/core/hle/service/am/service/application_functions.cpp` in each); libnx
        /// `appletGetDesiredLanguage` sends 21 to `IApplicationFunctions` with
        /// `_appletCmdNoInOutU64`. The scoping doc has no row for it.
        pub const GET_DESIRED_LANGUAGE: u32 = 21;
        /// `{210, D<&IApplicationFunctions::GetUnknownEvent210>, "Unknown210"}`, line 117 of
        /// `src/core/hle/service/am/service/application_functions.cpp` in Eden @ cf654052,
        /// Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97 alike; Eden's name. libnx has no
        /// client for it, and the scoping doc has no row for it.
        pub const GET_UNKNOWN_EVENT_210: u32 = 210;
    }

    /// `IGraphicsScalingParametersFunctions`' commands (Switch 2), from Eden PR #4588
    /// (`eden-pr4588/src/core/hle/service/am/service/graphics_scaling_parameters_functions.cpp`
    /// @ e5fb3b97, `IGraphicsScalingParametersFunctions::IGraphicsScalingParametersFunctions`).
    pub mod graphics_scaling_parameters_functions {
        /// `{10, D<&IGraphicsScalingParametersFunctions::GetGraphicsScalingParameters>, ..}`.
        pub const GET_GRAPHICS_SCALING_PARAMETERS: u32 = 10;
        /// `{1050,
        /// D<&IGraphicsScalingParametersFunctions::SetGraphicsScaleChangedNotificationEnabled>,
        /// ..}`.
        pub const SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED: u32 = 1050;
    }
}

use command::{
    application_functions, common_state_getter, graphics_scaling_parameters_functions, proxy,
    proxy_service, self_controller, window_controller,
};

commands! {
    impl IApplicationProxyService {
        proxy_service::OPEN_APPLICATION_PROXY => open_application_proxy,
        proxy_service::OPEN_APPLICATION_PROXY_WITH_FLAGS => open_application_proxy_with_flags,
    }
}

commands! {
    impl IApplicationProxy {
        proxy::GET_COMMON_STATE_GETTER => get_common_state_getter,
        proxy::GET_SELF_CONTROLLER => get_self_controller,
        proxy::GET_WINDOW_CONTROLLER => get_window_controller,
        proxy::GET_AUDIO_CONTROLLER => get_audio_controller,
        proxy::GET_DISPLAY_CONTROLLER => get_display_controller,
        proxy::GET_LIBRARY_APPLET_CREATOR => get_library_applet_creator,
        proxy::GET_APPLICATION_FUNCTIONS => get_application_functions,
        proxy::GET_PERFORMANCE_FUNCTIONS => get_performance_functions,
        proxy::GET_GRAPHICS_SCALING_PARAMETERS_FUNCTIONS =>
            get_graphics_scaling_parameters_functions,
        proxy::GET_DEBUG_FUNCTIONS => get_debug_functions,
    }
}

commands! {
    impl ICommonStateGetter {
        common_state_getter::GET_EVENT_HANDLE => get_event_handle,
        common_state_getter::RECEIVE_MESSAGE => receive_message,
        common_state_getter::GET_OPERATION_MODE => get_operation_mode,
        common_state_getter::GET_PERFORMANCE_MODE => get_performance_mode,
        common_state_getter::GET_CURRENT_FOCUS_STATE => get_current_focus_state,
        common_state_getter::SET_CPU_BOOST_MODE => set_cpu_boost_mode,
        common_state_getter::SET_REQUEST_EXIT_TO_LIBRARY_APPLET_AT_EXECUTE_NEXT_PROGRAM_ENABLED =>
            set_request_exit_to_library_applet_at_execute_next_program_enabled,
    }
}

commands! {
    impl ISelfController {
        self_controller::EXIT => exit,
        self_controller::SET_SCREEN_SHOT_PERMISSION => set_screen_shot_permission,
        self_controller::SET_OPERATION_MODE_CHANGED_NOTIFICATION =>
            set_operation_mode_changed_notification,
        self_controller::SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION =>
            set_performance_mode_changed_notification,
        self_controller::SET_FOCUS_HANDLING_MODE => set_focus_handling_mode,
        self_controller::SET_RESTART_MESSAGE_ENABLED => set_restart_message_enabled,
        self_controller::SET_OUT_OF_FOCUS_SUSPENDING_ENABLED =>
            set_out_of_focus_suspending_enabled,
        self_controller::GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT =>
            get_accumulated_suspended_tick_changed_event,
        self_controller::GET_GRAPHICS_SUPERVISOR => get_graphics_supervisor,
        self_controller::SET_RTC_SCREEN_SHARING_AUDIO_ENABLED =>
            set_rtc_screen_sharing_audio_enabled,
    }
}

commands! {
    impl IWindowController {
        window_controller::GET_APPLET_RESOURCE_USER_ID => get_applet_resource_user_id,
        window_controller::ACQUIRE_FOREGROUND_RIGHTS => acquire_foreground_rights,
    }
}

commands! {
    impl IApplicationFunctions {
        application_functions::GET_DESIRED_LANGUAGE => get_desired_language,
        application_functions::NOTIFY_RUNNING => notify_running,
        application_functions::IS_GAME_PLAY_RECORDING_SUPPORTED =>
            is_game_play_recording_supported,
        application_functions::SET_GAME_PLAY_RECORDING_STATE => set_game_play_recording_state,
        application_functions::GET_GPU_ERROR_DETECTED_SYSTEM_EVENT =>
            get_gpu_error_detected_system_event,
        application_functions::GET_UNKNOWN_EVENT_210 => get_unknown_event_210,
        application_functions::GET_SCREEN_SHOT_AND_CONTINUOUS_RECORDING_INITIALLY_ENABLED =>
            get_screen_shot_and_continuous_recording_initially_enabled,
        application_functions::UNKNOWN_321 => unknown_321,
        application_functions::UNKNOWN_330 => unknown_330,
    }
}

commands! {
    impl IGraphicsScalingParametersFunctions {
        graphics_scaling_parameters_functions::GET_GRAPHICS_SCALING_PARAMETERS =>
            get_graphics_scaling_parameters,
        graphics_scaling_parameters_functions::SET_GRAPHICS_SCALE_CHANGED_NOTIFICATION_ENABLED =>
            set_graphics_scale_changed_notification_enabled,
    }
}

// No command of these five is ported (module docs, "Commands"); every request to one goes to the
// unknown-command policy.
commands! {
    impl IAudioController {}
}

commands! {
    impl IDisplayController {}
}

commands! {
    impl ILibraryAppletCreator {}
}

commands! {
    impl IDebugFunctions {}
}

commands! {
    impl IPerformanceFunctions {}
}

impl IApplicationProxyService {
    /// `OpenApplicationProxy`. in: the caller's process id, and its process as copy handle 0.
    /// out: an [`IApplicationProxy`].
    fn open_application_proxy(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::open_proxy(req, reply)
    }

    /// `OpenApplicationProxyWithFlags` (Switch 2). in: `u32` flags, then as
    /// `OpenApplicationProxy`. out: an [`IApplicationProxy`]. The flags are checked first
    /// ([`PROXY_FLAGS_ACCEPTED`]), as PR #4588 does before it calls `OpenApplicationProxy`.
    fn open_application_proxy_with_flags(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let flags = req.payload().read_u32()?;
        if !PROXY_FLAGS_ACCEPTED.contains(&flags) {
            tracing::debug!(
                flags,
                "appletOE: OpenApplicationProxyWithFlags refuses its flags"
            );
            return Err(UNKNOWN.into());
        }
        Self::open_proxy(req, reply)
    }

    /// What `OpenApplicationProxy` does, and `OpenApplicationProxyWithFlags` once its flags are
    /// accepted: checks the process id, then the process handle, then returns the proxy.
    fn open_proxy(req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let Some(aruid) = req.process_id() else {
            return Err(UNKNOWN.into());
        };
        req.copy_handle(0)?;
        reply.push_session(Box::new(IApplicationProxy { aruid }));
        reply.ok()
    }
}

impl IApplicationProxy {
    /// `GetCommonStateGetter`. out: an [`ICommonStateGetter`].
    fn get_common_state_getter(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ICommonStateGetter));
        reply.ok()
    }

    /// `GetSelfController`. out: an [`ISelfController`].
    fn get_self_controller(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ISelfController));
        reply.ok()
    }

    /// `GetWindowController`. out: an [`IWindowController`].
    fn get_window_controller(
        &mut self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IWindowController { aruid: self.aruid }));
        reply.ok()
    }

    /// `GetAudioController`. out: an [`IAudioController`].
    fn get_audio_controller(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IAudioController));
        reply.ok()
    }

    /// `GetDisplayController`. out: an [`IDisplayController`].
    fn get_display_controller(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IDisplayController));
        reply.ok()
    }

    /// `GetLibraryAppletCreator`. out: an [`ILibraryAppletCreator`].
    fn get_library_applet_creator(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ILibraryAppletCreator));
        reply.ok()
    }

    /// `GetApplicationFunctions`. out: an [`IApplicationFunctions`].
    fn get_application_functions(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IApplicationFunctions));
        reply.ok()
    }

    /// `GetPerformanceFunctions` (Switch 2). out: an [`IPerformanceFunctions`].
    fn get_performance_functions(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IPerformanceFunctions));
        reply.ok()
    }

    /// `GetGraphicsScalingParametersFunctions` (Switch 2). out: an
    /// [`IGraphicsScalingParametersFunctions`].
    fn get_graphics_scaling_parameters_functions(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IGraphicsScalingParametersFunctions));
        reply.ok()
    }

    /// `GetDebugFunctions`. out: an [`IDebugFunctions`].
    fn get_debug_functions(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IDebugFunctions));
        reply.ok()
    }
}

impl ICommonStateGetter {
    /// `GetEventHandle`. out: a copy handle to the applet's message event.
    fn get_event_handle(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply_with_applet_event(cx, reply, Applet::message_event_or_create)
    }

    /// `ReceiveMessage`. out: `u32` the applet's next message ([`AppletMessage::raw`]), or
    /// [`NO_MESSAGES`] with nothing when there is none (Eden: `if
    /// (!lifecycle_manager.PopMessage(..)) R_THROW(AM::ResultNoMessages)`).
    ///
    /// The message event is cleared (if no message will be left) before the message is taken:
    /// a clear that fails (a restored event id that is not an event) answers the kernel's error
    /// and leaves the message to be received again ([`Applet::pop_message`]; module docs, "Where
    /// this differs from Eden").
    fn receive_message(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let message = Applet::pop_message(cx)?;
        let Some(message) = message else {
            tracing::debug!("appletOE: ReceiveMessage with no message");
            return Err(NO_MESSAGES.into());
        };
        tracing::debug!(message = message.raw(), "appletOE: ReceiveMessage");
        reply.payload().write_u32(message.raw());
        reply.ok()
    }

    /// `SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled`. in: nothing. out: nothing.
    /// Sets the applet's flag (Eden: `m_applet->
    /// request_exit_to_library_applet_at_execute_next_program_enabled = true`), which no
    /// reference reads.
    fn set_request_exit_to_library_applet_at_execute_next_program_enabled(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        cx.family::<Applet>()?
            .request_exit_to_library_applet_at_execute_next_program_enabled = true;
        reply.ok()
    }

    /// `SetCpuBoostMode`. in: `u32` mode. out: nothing. Always success, as Eden's (`APM_Sys::
    /// SetCpuBoostMode` answers `ResultSuccess` whatever the mode). The mode is checked as
    /// Eden's `Controller::SetFromCpuBoostMode` checks it ([`CpuBoostMode::from_raw`]), which
    /// changes only what is logged: Eden records a performance configuration for a mode it
    /// accepts, which nothing here keeps (module docs, "The title's later start-up commands").
    fn set_cpu_boost_mode(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mode = req.payload().read_u32()?;
        if CpuBoostMode::from_raw(mode).is_some() {
            tracing::debug!(mode, "appletOE: SetCpuBoostMode");
        } else {
            tracing::warn!(
                mode,
                "appletOE: SetCpuBoostMode with a word that is not a mode"
            );
        }
        reply.ok()
    }

    /// `GetOperationMode`. out: `u8` operation mode.
    fn get_operation_mode(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mode = cx.family::<Applet>()?.settings.operation_mode;
        reply.payload().write_u8(mode.raw());
        reply.ok()
    }

    /// `GetPerformanceMode`. out: `s32` performance mode.
    fn get_performance_mode(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mode = cx.family::<Applet>()?.settings.performance_mode();
        reply.payload().write_i32(mode.raw());
        reply.ok()
    }

    /// `GetCurrentFocusState`. out: `u8` focus state.
    ///
    /// Eden's `GetAndClearFocusState` also records the state as acknowledged, which only a
    /// library applet's messages read (`PopMessageInOrderOfPriority`, `!m_is_application`), so
    /// an application's applet does not keep it.
    fn get_current_focus_state(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let state = cx.family::<Applet>()?.settings.focus_state;
        reply.payload().write_u8(state.raw());
        reply.ok()
    }
}

impl ISelfController {
    /// Runs `change` on the applet's lifecycle state, signalling the message event as
    /// [`Applet::update`] says.
    fn update(cx: &mut Context<'_>, change: impl FnOnce(&mut Lifecycle)) -> Result<(), Fault> {
        Applet::update(cx, change)
    }

    /// `Exit`: the caller's process ends once this success reply reaches it (module docs,
    /// "Where this differs from Eden"). Eden: `m_applet->process->Terminate()`, then
    /// `R_SUCCEED()`.
    fn exit(_: &mut Self, cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        tracing::info!("appletOE: the application exits");
        cx.end_process()?;
        reply.ok()
    }

    /// `SetOperationModeChangedNotification`. in: `bool` enabled.
    fn set_operation_mode_changed_notification(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        Self::update(cx, |lifecycle| {
            lifecycle.notifications.operation_mode_changed = enabled;
        })?;
        reply.ok()
    }

    /// `SetPerformanceModeChangedNotification`. in: `bool` enabled.
    fn set_performance_mode_changed_notification(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        Self::update(cx, |lifecycle| {
            lifecycle.notifications.performance_mode_changed = enabled;
        })?;
        reply.ok()
    }

    /// `SetFocusHandlingMode`. in: `bool notify`, `bool background` (not read, as in Eden, which
    /// only logs it), `bool suspend`.
    ///
    /// Eden: `SetFocusStateChangedNotificationEnabled(notify)` (which signals if needed), then
    /// `SetFocusHandlingMode(suspend)`, then `UpdateSuspensionStateLocked(true)`, which signals
    /// if needed again; nothing between the two checks changes whether a message is available,
    /// so the second finds nothing to do.
    fn set_focus_handling_mode(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let notify = read_flag(&mut payload)?;
        let _background = read_flag(&mut payload)?;
        let suspend = read_flag(&mut payload)?;
        Self::update(cx, |lifecycle| {
            lifecycle.notifications.focus_state_changed = notify;
            lifecycle.set_focus_handling_mode(suspend);
        })?;
        reply.ok()
    }

    /// `SetOutOfFocusSuspendingEnabled`. in: `bool` enabled.
    ///
    /// Eden's `UpdateSuspensionStateLocked(false)` after it signals only when the focus state or
    /// whether the process may run changed, which nothing in a run changes.
    fn set_out_of_focus_suspending_enabled(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        Self::update(cx, |lifecycle| {
            lifecycle.set_out_of_focus_suspending_enabled(enabled);
        })?;
        reply.ok()
    }

    /// `GetAccumulatedSuspendedTickChangedEvent`. out: a copy handle to the applet's
    /// suspended-tick event.
    fn get_accumulated_suspended_tick_changed_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply_with_applet_event(cx, reply, Applet::suspended_tick_event_or_create)
    }

    /// `SetScreenShotPermission`. in: `u32` permission, any value, kept in the applet (Eden PR
    /// #4588: `m_applet->screenshot_permission = screen_shot_permission`). out: nothing.
    fn set_screen_shot_permission(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let permission = ScreenshotPermission::from_raw(req.payload().read_u32()?);
        cx.family::<Applet>()?.screenshot_permission = permission;
        reply.ok()
    }

    /// `SetRestartMessageEnabled`. in: `bool` enabled, kept in the applet
    /// ([`Applet::resume_notification_enabled`]). out: nothing. Eden:
    /// `lifecycle_manager.SetResumeNotificationEnabled(enabled)`, which only sets the flag: it
    /// does not call `SignalSystemEventIfNeeded`, and the flag is not a term of whether a
    /// message is available, so the message event is not touched.
    fn set_restart_message_enabled(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        cx.family::<Applet>()?.resume_notification_enabled = enabled;
        reply.ok()
    }

    /// `SetRtcScreenSharingAudioEnabled` (Switch 2). in: `bool` enabled, not kept. out:
    /// nothing. PR #4588 only logs it ("SDK 23.0 serializes this command as one bool input
    /// with no output payload").
    fn set_rtc_screen_sharing_audio_enabled(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        tracing::debug!(enabled, "appletOE: SetRtcScreenSharingAudioEnabled");
        reply.ok()
    }

    /// `GetGraphicsSupervisor` (Switch 2). out: a copy handle to a new session on the graphics
    /// server's root (module docs, "The graphics server").
    fn get_graphics_supervisor(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(root) = cx.family::<Applet>()?.graphics_server else {
            tracing::warn!("appletOE: GetGraphicsSupervisor with no graphics server in the run");
            return Err(UNKNOWN.into());
        };
        reply.copy_service(root);
        reply.ok()
    }
}

impl IWindowController {
    /// `GetAppletResourceUserId`. out: `u64` applet resource user id.
    fn get_applet_resource_user_id(
        &mut self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u64(self.aruid);
        reply.ok()
    }

    /// `AcquireForegroundRights`. Eden only logs.
    fn acquire_foreground_rights(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.ok()
    }
}

impl IApplicationFunctions {
    /// `NotifyRunning`. out: `bool` became running, always true (Eden: `*out_became_running =
    /// true`).
    fn notify_running(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u8(u8::from(true));
        reply.ok()
    }

    /// `GetDesiredLanguage`. out: `u64` language code, [`DESIRED_LANGUAGE_CODE`].
    fn get_desired_language(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u64(DESIRED_LANGUAGE_CODE);
        reply.ok()
    }

    /// `GetUnknownEvent210`. out: a copy handle to the applet's unknown event (Eden: `*out_event
    /// = m_applet->unknown_event.GetHandle()`), made the first time and never signalled.
    fn get_unknown_event_210(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply_with_applet_event(cx, reply, Applet::unknown_event_or_create)
    }

    /// `IsGamePlayRecordingSupported`. out: `bool`, [`GAME_PLAY_RECORDING_SUPPORTED`] (Eden PR
    /// #4588: `*out_is_game_play_recording_supported = m_applet->game_play_recording_supported`).
    fn is_game_play_recording_supported(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply
            .payload()
            .write_u8(u8::from(GAME_PLAY_RECORDING_SUPPORTED));
        reply.ok()
    }

    /// `SetGamePlayRecordingState`. in: `u32` state, any value, kept in the applet (Eden PR
    /// #4588: `m_applet->game_play_recording_state = game_play_recording_state`). out: nothing.
    fn set_game_play_recording_state(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let state = GamePlayRecordingState::from_raw(req.payload().read_u32()?);
        cx.family::<Applet>()?.game_play_recording_state = state;
        reply.ok()
    }

    /// `GetGpuErrorDetectedSystemEvent`. out: a copy handle to the applet's GPU-error event
    /// (Eden: `*out_event = m_applet->gpu_error_detected_event.GetHandle()`), made the first
    /// time and never signalled.
    fn get_gpu_error_detected_system_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply_with_applet_event(cx, reply, Applet::gpu_error_detected_event_or_create)
    }

    /// `GetScreenShotAndContinuousRecordingInitiallyEnabled` (Switch 2). out: `bool screenshot,
    /// bool recording`, a byte each, from the applet
    /// ([`Applet::screenshot_and_continuous_recording_initially_enabled`]).
    fn get_screen_shot_and_continuous_recording_initially_enabled(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let (screenshot, recording) = cx
            .family::<Applet>()?
            .screenshot_and_continuous_recording_initially_enabled();
        reply
            .payload()
            .write_u8(u8::from(screenshot))
            .write_u8(u8::from(recording));
        reply.ok()
    }

    /// `Unknown321` (Switch 2). out: `bool`, [`UNKNOWN_321_ENABLED`].
    fn unknown_321(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u8(u8::from(UNKNOWN_321_ENABLED));
        reply.ok()
    }

    /// `Unknown330`. out: `u8`, [`UNKNOWN_330_VALUE`].
    fn unknown_330(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u8(UNKNOWN_330_VALUE);
        reply.ok()
    }
}

impl IGraphicsScalingParametersFunctions {
    /// `GetGraphicsScalingParameters`. out: [`GraphicsScalingParameters::PR_4588`].
    fn get_graphics_scaling_parameters(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        GraphicsScalingParameters::PR_4588.write(reply.payload());
        reply.ok()
    }

    /// `SetGraphicsScaleChangedNotificationEnabled`. in: `bool` enabled, kept in the applet.
    fn set_graphics_scale_changed_notification_enabled(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let enabled = read_flag(&mut req.payload())?;
        cx.family::<Applet>()?
            .graphics_scale_changed_notification_enabled = enabled;
        reply.ok()
    }
}

/// The interfaces an `appletOE` server's sessions can be.
#[must_use]
pub fn oe_interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IApplicationProxyService>()
        .with::<IApplicationProxy>()
        .with::<ICommonStateGetter>()
        .with::<ISelfController>()
        .with::<IWindowController>()
        .with::<IApplicationFunctions>()
        .with::<IAudioController>()
        .with::<IDisplayController>()
        .with::<ILibraryAppletCreator>()
        .with::<IDebugFunctions>()
        .with::<IPerformanceFunctions>()
        .with::<IGraphicsScalingParametersFunctions>()
}

/// The `appletOE` server, for an application that reports `settings`, in a run without a
/// graphics server.
#[must_use]
pub fn oe_server(settings: AppletSettings, policy: UnknownPolicy) -> Server {
    oe_server_for(Applet::new(settings), policy)
}

/// The `appletOE` server for `applet`: what [`oe_server`] builds, for an applet made with
/// [`Applet::new`] and, in a run with a graphics server, [`Applet::with_graphics_server`].
#[must_use]
pub fn oe_server_for(applet: Applet, policy: UnknownPolicy) -> Server {
    Server::new(OE_NAME, policy)
        .with_root(|| Box::new(IApplicationProxyService))
        .with_interfaces(oe_interfaces())
        .with_checked_family(applet, same_graphics_server)
}

/// The restore check of the `appletOE` server's applet: a snapshot's graphics server number
/// is the run's (module docs, "The graphics server").
fn same_graphics_server(run: &Applet, saved: &Applet) -> Result<(), &'static str> {
    if run.graphics_server == saved.graphics_server {
        Ok(())
    } else {
        Err("its applet names another graphics server")
    }
}
