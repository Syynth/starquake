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
//! `applet_manager.cpp` (the launch) and `am_types.h` (`FocusState`, `OperationMode`). The request
//! layouts are checked against the client, libnx (`libnx/nx/source/services/applet.c`, ISC).
//!
//! # Commands
//!
//! Ids and names are Eden's tables. These are exactly the commands the startup fixture's trace
//! sends (`tests/homebrew/startup/reference/startup.trace`; its README's gap list named them);
//! every other command of these interfaces is left out and answered by the server's
//! unknown-command policy.
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `appletOE` ([`IApplicationProxyService`]) | 0 | `OpenApplicationProxy` | the caller's process id, attached by the kernel; a `u64` placeholder in the payload, not read; copy handle 0: the caller's process | an [`IApplicationProxy`] |
//! | [`IApplicationProxy`] | 0 | `GetCommonStateGetter` | nothing | an [`ICommonStateGetter`] |
//! | | 1 | `GetSelfController` | nothing | an [`ISelfController`] |
//! | | 2 | `GetWindowController` | nothing | an [`IWindowController`] |
//! | | 3 | `GetAudioController` | nothing | an [`IAudioController`] |
//! | | 4 | `GetDisplayController` | nothing | an [`IDisplayController`] |
//! | | 11 | `GetLibraryAppletCreator` | nothing | an [`ILibraryAppletCreator`] |
//! | | 20 | `GetApplicationFunctions` | nothing | an [`IApplicationFunctions`] |
//! | | 1000 | `GetDebugFunctions` | nothing | an [`IDebugFunctions`] |
//! | [`ICommonStateGetter`] | 0 | `GetEventHandle` | nothing | copy handle: the applet's message event |
//! | | 5 | `GetOperationMode` | nothing | `u8` [`OperationMode`] |
//! | | 6 | `GetPerformanceMode` | nothing | `s32` [`PerformanceMode`] |
//! | | 9 | `GetCurrentFocusState` | nothing | `u8` [`FocusState`] |
//! | [`ISelfController`] | 11 | `SetOperationModeChangedNotification` | `bool` (a byte) | nothing |
//! | | 12 | `SetPerformanceModeChangedNotification` | `bool` | nothing |
//! | | 13 | `SetFocusHandlingMode` | `bool notify, bool background, bool suspend`, a byte each | nothing |
//! | | 16 | `SetOutOfFocusSuspendingEnabled` | `bool` | nothing |
//! | | 91 | `GetAccumulatedSuspendedTickChangedEvent` | nothing | copy handle: the applet's suspended-tick event |
//! | [`IWindowController`] | 1 | `GetAppletResourceUserId` | nothing | `u64` the applet resource user id |
//! | | 10 | `AcquireForegroundRights` | nothing | nothing |
//! | [`IApplicationFunctions`] | 40 | `NotifyRunning` | nothing | `bool` (a byte): true |
//!
//! [`IAudioController`], [`IDisplayController`], [`ILibraryAppletCreator`] and
//! [`IDebugFunctions`] have no command here: libnx opens them at start-up and sends them nothing.
//! `ISelfController::Exit` (0), which libnx sends when the program ends, is not ported: the
//! fixture's run stops at `hid` well before it, and ending the process needs a kernel interface
//! the services spec does not give a service.
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
//! - `GetEventHandle`, `GetAccumulatedSuspendedTickChangedEvent`: libnx `_appletCmdGetEvent` (one
//!   copied handle); Eden `OutCopyHandle<KReadableEvent>`.
//! - `GetOperationMode`, `GetCurrentFocusState`: libnx `_appletCmdNoInOutU8`; Eden
//!   `Out<OperationMode>` and `Out<FocusState>`, both `u8` enums (`am_types.h`).
//!   `GetPerformanceMode`: libnx `_appletCmdNoInOutU32`; Eden `Out<APM::PerformanceMode>`, an
//!   `s32` enum.
//! - `GetAppletResourceUserId`: libnx `_appletCmdNoInOutU64`; Eden `Out<AppletResourceUserId>`,
//!   a `ProcessId` (`eden/src/core/hle/service/cmif_types.h`), one `u64`.
//! - `NotifyRunning`: libnx `_appletCmdNoInOutBool` (a `u8`, of which it keeps bit 0); Eden
//!   `Out<bool>`, set to true.
//! - The `Set...` commands with one flag: libnx `_appletCmdInBoolNoOut` (one `u8`, 0 or 1); Eden a
//!   `bool` argument. `SetFocusHandlingMode`: libnx `_appletSetFocusHandlingMode`
//!   (`struct { u8 inval0; u8 inval1; u8 inval2; }`); Eden `(bool notify, bool background, bool
//!   suspend)`, at offsets 0, 1 and 2 (`cmif_serialization.h`, `ReadInArgument`: each argument at
//!   its own alignment).
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
//! # The message event
//!
//! `GetEventHandle` returns the applet's message event (Eden: the lifecycle manager's system
//! event), which is signalled while the applet has a message to receive. Of Eden's
//! `ShouldSignalSystemEvent`, this port keeps the one term a running application can reach:
//! focus-state-change notifications are enabled and the focus state has changed since the
//! applet last received a message. Its other terms (unordered messages, resume, exit,
//! sleep, display requests, operation- and performance-mode changes, SD card removal,
//! temperature, battery, auto power-down, album events, and the library-applet focus terms) are
//! all false in a run: nothing here raises them. So the event is signalled from launch (Eden's
//! `SetFocusState(InFocus)` at launch finds a focus change pending, with notifications on), and
//! libnx's `SetFocusHandlingMode(0, 0, 1)` turns the notifications off.
//!
//! Eden signals or clears the event in `SignalSystemEventIfNeeded`, when the cached "a message
//! is available" differs from `ShouldSignalSystemEvent`. Here the event's state is compared
//! before and after each command that changes an input of that condition ([`Lifecycle`]): the
//! same transitions, because in this port only those commands change the inputs, and Eden calls
//! `SignalSystemEventIfNeeded` after each of them.
//!
//! The events are created the first time a command returns them, not when the applet is: the
//! kernel interface is reached only while a request is handled. A message event created then is
//! signalled exactly when the condition above holds, which is the state Eden's event is in at
//! that moment. The suspended-tick event is never signalled, as in Eden (only its constructor
//! names it, `applet.cpp`).
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `am/tests.rs`:
//!
//! - **The message event is never cleared.** Where Eden's `SignalSystemEventIfNeeded` clears it
//!   (a message stops being available: libnx's `SetFocusHandlingMode(0, 0, 1)` at start-up), the
//!   kernel event here stays signalled, because the services spec's kernel interface creates and
//!   signals events and does not clear them. The applet's own record of the condition does go
//!   false, so a later change signals the event again as Eden would
//!   (`turning_focus_notifications_off_does_not_clear_the_message_event`). A guest that waits on
//!   the event then wakes where on Eden it would sleep; `ReceiveMessage`, which it would send
//!   next, is not ported.
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
//! The Switch 2 additions (`appletOE` command 10, `IApplicationProxy` 40 and 60,
//! `IApplicationFunctions` 320 and 321, `ISelfController` 2000:
//! `docs/switch2-eden-diff-scoping.md`, "Services (HLE)") are left out; each is one more row
//! and method.
//!
//! [`INVALID_CMIF_REQUEST`]: crate::framework::result::INVALID_CMIF_REQUEST

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use crate::apm::PerformanceMode;
use crate::commands;
use crate::framework::{
    Context, Fault, Interfaces, Outcome, Reader, Reply, Request, Server, UnknownPolicy,
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

/// The part of Eden's `LifecycleManager` that the ported commands read and change, with Eden's
/// initial values (`lifecycle_manager.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lifecycle {
    /// `m_focus_state_changed_notification_enabled`, initially true.
    pub focus_state_changed_notification_enabled: bool,
    /// `m_operation_mode_changed_notification_enabled`, initially true.
    pub operation_mode_changed_notification_enabled: bool,
    /// `m_performance_mode_changed_notification_enabled`, initially true.
    pub performance_mode_changed_notification_enabled: bool,
    /// `m_has_focus_state_changed`, initially true; only `ReceiveMessage` (not ported) clears it.
    pub has_focus_state_changed: bool,
    /// `m_focus_handling_mode`.
    pub focus_handling_mode: FocusHandlingMode,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            focus_state_changed_notification_enabled: true,
            operation_mode_changed_notification_enabled: true,
            performance_mode_changed_notification_enabled: true,
            has_focus_state_changed: true,
            focus_handling_mode: FocusHandlingMode::SuspendHomeSleep,
        }
    }
}

impl Lifecycle {
    /// Whether the applet has a message to receive, so its message event is signalled: the one
    /// term of `LifecycleManager::ShouldSignalSystemEvent` a running application can reach (module
    /// docs, "The message event").
    #[must_use]
    pub const fn message_available(&self) -> bool {
        self.focus_state_changed_notification_enabled && self.has_focus_state_changed
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
/// A method that changes it while it calls the kernel works on a copy and writes the copy back
/// before it reports any error, so a kernel object it created is never forgotten.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Applet {
    settings: AppletSettings,
    lifecycle: Lifecycle,
    /// The message event, once a command has returned it.
    message_event: Option<ObjectId>,
    /// The suspended-tick event, once a command has returned it.
    suspended_tick_event: Option<ObjectId>,
}

impl Applet {
    /// A freshly launched application's applet, reporting `settings`.
    #[must_use]
    pub fn new(settings: AppletSettings) -> Self {
        Self {
            settings,
            lifecycle: Lifecycle::default(),
            message_event: None,
            suspended_tick_event: None,
        }
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

    /// Changes the lifecycle state with `change`, then signals the message event if a message
    /// became available: Eden's `SignalSystemEventIfNeeded` after the change (module docs, "The
    /// message event"). Where a message stops being available Eden clears the event; that is
    /// the listed divergence, and only the state records it.
    fn update(
        &mut self,
        cx: &mut Context<'_>,
        change: impl FnOnce(&mut Lifecycle),
    ) -> Result<(), Fault> {
        let before = self.lifecycle.message_available();
        change(&mut self.lifecycle);
        let after = self.lifecycle.message_available();
        if after
            && !before
            && let Some(event) = self.message_event
        {
            cx.signal_event(event)?;
        }
        if before && !after {
            tracing::debug!(
                "appletOE: the message event would be cleared here; it stays signalled"
            );
        }
        Ok(())
    }

    /// The message event, created (and signalled if a message is available) the first time.
    fn message_event_or_create(&mut self, cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        if let Some(event) = self.message_event {
            return Ok(event);
        }
        let event = cx.create_event()?;
        // Kept before it is signalled, so the event the server holds a reference to is never
        // lost; signalling an event just created fails only if the kernel lost it.
        self.message_event = Some(event);
        if self.lifecycle.message_available() {
            cx.signal_event(event)?;
        }
        Ok(event)
    }

    /// The suspended-tick event, created the first time. It is never signalled.
    fn suspended_tick_event_or_create(&mut self, cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        if let Some(event) = self.suspended_tick_event {
            return Ok(event);
        }
        let event = cx.create_event()?;
        self.suspended_tick_event = Some(event);
        Ok(event)
    }
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

/// The ids of the commands ported here, per interface (Eden's tables, each row cited).
pub mod command {
    /// `appletOE`'s commands (`IApplicationProxyService::IApplicationProxyService`).
    pub mod proxy_service {
        /// `{0, D<&IApplicationProxyService::OpenApplicationProxy>, "OpenApplicationProxy"}`.
        pub const OPEN_APPLICATION_PROXY: u32 = 0;
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
        /// `{1000, D<&IApplicationProxy::GetDebugFunctions>, "GetDebugFunctions"}`.
        pub const GET_DEBUG_FUNCTIONS: u32 = 1000;
    }

    /// `ICommonStateGetter`'s commands (`ICommonStateGetter::ICommonStateGetter`).
    pub mod common_state_getter {
        /// `{0, D<&ICommonStateGetter::GetEventHandle>, "GetEventHandle"}`.
        pub const GET_EVENT_HANDLE: u32 = 0;
        /// `{5, D<&ICommonStateGetter::GetOperationMode>, "GetOperationMode"}`.
        pub const GET_OPERATION_MODE: u32 = 5;
        /// `{6, D<&ICommonStateGetter::GetPerformanceMode>, "GetPerformanceMode"}`.
        pub const GET_PERFORMANCE_MODE: u32 = 6;
        /// `{9, D<&ICommonStateGetter::GetCurrentFocusState>, "GetCurrentFocusState"}`.
        pub const GET_CURRENT_FOCUS_STATE: u32 = 9;
    }

    /// `ISelfController`'s commands (`ISelfController::ISelfController`).
    pub mod self_controller {
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
    }
}

use command::{
    application_functions, common_state_getter, proxy, proxy_service, self_controller,
    window_controller,
};

commands! {
    impl IApplicationProxyService {
        proxy_service::OPEN_APPLICATION_PROXY => open_application_proxy,
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
        proxy::GET_DEBUG_FUNCTIONS => get_debug_functions,
    }
}

commands! {
    impl ICommonStateGetter {
        common_state_getter::GET_EVENT_HANDLE => get_event_handle,
        common_state_getter::GET_OPERATION_MODE => get_operation_mode,
        common_state_getter::GET_PERFORMANCE_MODE => get_performance_mode,
        common_state_getter::GET_CURRENT_FOCUS_STATE => get_current_focus_state,
    }
}

commands! {
    impl ISelfController {
        self_controller::SET_OPERATION_MODE_CHANGED_NOTIFICATION =>
            set_operation_mode_changed_notification,
        self_controller::SET_PERFORMANCE_MODE_CHANGED_NOTIFICATION =>
            set_performance_mode_changed_notification,
        self_controller::SET_FOCUS_HANDLING_MODE => set_focus_handling_mode,
        self_controller::SET_OUT_OF_FOCUS_SUSPENDING_ENABLED =>
            set_out_of_focus_suspending_enabled,
        self_controller::GET_ACCUMULATED_SUSPENDED_TICK_CHANGED_EVENT =>
            get_accumulated_suspended_tick_changed_event,
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
        application_functions::NOTIFY_RUNNING => notify_running,
    }
}

// No command of these four is ported (module docs, "Commands"); every request to one goes to the
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

impl IApplicationProxyService {
    /// `OpenApplicationProxy`. in: the caller's process id, and its process as copy handle 0.
    /// out: an [`IApplicationProxy`].
    fn open_application_proxy(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
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
        let mut applet = *cx.family::<Applet>()?;
        let event = applet.message_event_or_create(cx);
        *cx.family::<Applet>()? = applet;
        reply.copy_handle(event?);
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
        let mut applet = *cx.family::<Applet>()?;
        let updated = applet.update(cx, change);
        *cx.family::<Applet>()? = applet;
        updated
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
            lifecycle.operation_mode_changed_notification_enabled = enabled;
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
            lifecycle.performance_mode_changed_notification_enabled = enabled;
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
            lifecycle.focus_state_changed_notification_enabled = notify;
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
        let mut applet = *cx.family::<Applet>()?;
        let event = applet.suspended_tick_event_or_create(cx);
        *cx.family::<Applet>()? = applet;
        reply.copy_handle(event?);
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
}

/// The `appletOE` server, for an application that reports `settings`.
#[must_use]
pub fn oe_server(settings: AppletSettings, policy: UnknownPolicy) -> Server {
    Server::new(OE_NAME, policy)
        .with_root(|| Box::new(IApplicationProxyService))
        .with_interfaces(oe_interfaces())
        .with_family(Applet::new(settings))
}
