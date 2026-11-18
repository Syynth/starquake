//! The vi2 session gsv 305 returns: displays, clients and their layers, request groups and the
//! display monitors registered on them, notifications, and display state
//! (`docs/specs/graphics-server.md`, "Display: vi2"; port slice 13). Present (228, 229) and its
//! pacing are slice 14's: until it lands they go to the server's unknown-command policy.
//!
//! * [`saved`]: the session's saved form and the relations a restore checks.
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp`
//!   @ e5fb3b97 (`IVi2Session`) and `graphics_supervisor_protocol.h` @ e5fb3b97
//!   (`RequestGroupCount`, `DecodeRequestGroupEventDuplicate`,
//!   `ClassifyRequestGroupEventDuplicate`, `IsSupportedDisplayParameter301`). PR names the
//!   commands from the vi2 system module (`vi2.autogen` addresses in its comments).
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`INx2Vi2Session`), and its blocker log `eden-nx2/NX2_NOTES.md` @ 50fa86bb, whose rows
//!   "row N" names.
//!
//! The VI result module and its description 7 and 5 are Eden's
//! (`eden/src/core/hle/result.h` @ cf654052: `ErrorModule::VI = 114`;
//! `eden/src/core/hle/service/vi/vi_results.h` @ cf654052: `ResultNotFound{VI, 7}`,
//! `ResultPermissionDenied{VI, 5}`); description 4 is PR's (`Result{ErrorModule::VI, 4}`), which
//! no reference names ([`VI_OUT_OF_RANGE`] is starquake's name for it).
//!
//! # The server
//!
//! [`vi2_server`] is the vi2 family's server (decision 6). Every session is a [`Vi2Session`],
//! and keeps its own state, as PR keeps everything per `IVi2Session` (`m_client_states`,
//! `m_request_group_events`, `m_notifications`, `m_display_references` and the rest). The
//! family ([`Vi2Family`]) holds only the wiring: gsv's shared state, whose vsync clock
//! ([`crate::shared::vsync`]) ticks the request groups. The root hands this server's sessions
//! out with 305 once it is given the server's number
//! ([`crate::root::RootState::with_vi2_server`]; [`crate::root`], "The display session (305)").
//!
//! The server reads the `Env` clock ([`Context::env`]) when a request group starts listening to
//! the vsync clock, which starts the clock one interval after that reading
//! ([`crate::shared::Locked::listen_vsync`]); a server built without an environment answers those
//! requests with the framework's `NOT_SUPPORTED`.
//!
//! # Commands
//!
//! Every command is TIPC (PR and NX2 `RegisterHandlersTipc`). Each reply is the result word,
//! then the data below. Words are counted from the start of the message: the two header words,
//! then the data. A display name is a `u64`, eight bytes of text, `"default"` being
//! [`DEFAULT_DISPLAY`]; a client is named by its `u64` token (PR: "the SDK sends its object
//! address as an opaque identity").
//!
//! | Id | Command (PR's name) | In | Out | Mark |
//! |---|---|---|---|---|
//! | 1 | `GetNotificationLimit` | nothing | `u32` [`NOTIFICATION_LIMIT`] | **conflict** (row 21; PR 0, NX2 1) |
//! | 3 | `GetSessionLimit` | nothing | `u32` [`SESSION_LIMIT`] | **conflict** (row 21; PR 0, NX2 1) |
//! | 5 | `GetRequestGroupLimit` | nothing | `u32` [`REQUEST_GROUP_LIMIT`] | observed (row 21; both 7) |
//! | 11 | `GetNotificationEvent` | `u32` 0 | copy handle: the notification event | read (PR); NX2 `ReturnEvent` |
//! | 13 | `ReleaseNotificationEvent` | `u32` 0 | nothing | read (PR) |
//! | 21 | `GetNotifications` | out buffer of at least [`NOTIFICATIONS_SIZE`] bytes | the bitmap, in the buffer | read (PR) |
//! | 31 | `CreateRequestGroupEvent` | `u32` group | copy handle: the group's event | observed (rows 21, 82); PR, **conflict** with NX2 (below) |
//! | 32 | `DuplicateRequestGroupEvent` | copy handle: the group's event; `u32` group | copy handle: the same event | observed (rows 21, 82); PR, **conflict** with NX2 (below) |
//! | 51 | `ConfigureRequestGroups` | `u32` count | nothing | read (PR) |
//! | 101 | `OpenDisplay` | `u64` name, `u32` flags | nothing | read (PR) |
//! | 102 | `CloseDisplay` | `u64` name | nothing | read (PR) |
//! | 104 | `GetHdrStatus` | `u64` name | `u64` format, `u32` 0 | read (PR); NX2 replies nothing (row 119) |
//! | 105 | `GetDisplayState105` | `u64` name | `u32` 0 | read (PR); NX2 replies nothing (row 119) |
//! | 106 | `GetBufferMaxResolution` | `u64` name | `u64` mode | read (PR) |
//! | 107 | `GetDisplayState107` | `u64` name | `u64` 1, `u64` 1, `u64` 0 | read (PR) |
//! | 111 | `OpenEndpoint` | `u64` name, `u64` display, `u64` client | nothing | read (PR) |
//! | 112 | `CloseEndpoint` | `u64` name, `u64` display | nothing | read (PR) |
//! | 201 | `RegisterClientState` | `u64` client, `u32` retire bit, `u32` available bit | nothing | read (PR) |
//! | 202 | `UnregisterClientState` | `u64` client | nothing | read (PR) |
//! | 212 | `GetClientAvailability` | `u64` client | `u64` mask of free slots | read (PR) |
//! | 251 | `BindClientDisplay` | `u64` client, `u64` name | nothing | read (PR) |
//! | 252 | `UnbindClientDisplay` | `u64` client | nothing | read (PR) |
//! | 261–272 | register (odd) and unregister (even) a display monitor | `u64` name, `u32` group | nothing | read (PR); kind 0 observed as the vsync (row 82) |
//! | 301 | `SetDisplayParameter301` | `u64` name, `u64` value, `u64` mask | nothing | read (PR) |
//! | 351 | `SetClientOutputEncoding` | `u64` client, `u64` encoding | nothing | read (PR) |
//! | 352 | `SetClientOutputLevels` | `u64` client, three `f32` | nothing | read (PR); **conflict** on the refused levels' result (below) |
//! | 354 | `GetDisplayState354` | `u64` name | `u64` 1 | read (PR) |
//!
//! What each checks, in order, answering the first failure (PR's checks, with PR's results but
//! where a bullet marks a **conflict**: 11, 13, 301 and 352):
//!
//! - **1, 3, 5** check nothing.
//! - **11** reads its word and takes any value: it makes the session's notification event the
//!   first time, signals it if any notification is raised, and hands it out. **13** reads its
//!   word, takes any value, and does nothing else. PR refuses a word other than 0 with
//!   `InvalidCombination` (`packet[2] == 0`); NX2 takes any (it logs `[2]`), and a title ran on
//!   that, so decision 8 takes NX2's (below, "Where this differs").
//! - **21** refuses an out buffer smaller than [`NOTIFICATIONS_SIZE`] with `InvalidSize`;
//!   otherwise it writes the bitmap ([`NOTIFICATION_WORDS`] little-endian `u32`s), clears the
//!   notification event if it has been made, and only then clears the bitmap, so a refused
//!   write or clear leaves every notification raised.
//! - **31** refuses a group at or past [`REQUEST_GROUPS`] with [`VI_OUT_OF_RANGE`], a group whose
//!   event exists with [`VI_PERMISSION_DENIED`], and a vsync clock with no room for another
//!   listener (below, "Vsync") with `OutOfResource`; otherwise it makes the group's event,
//!   which starts listening to the vsync clock, and hands it out.
//! - **32** reads the copied handle (a handle naming nothing is the kernel's error), then refuses
//!   a group at or past [`REQUEST_GROUPS`] with [`VI_OUT_OF_RANGE`], a group with no event with
//!   [`VI_NOT_FOUND`], and a handle naming another object than the group's event with
//!   `InvalidHandle` (PR `ClassifyRequestGroupEventDuplicate`); otherwise it hands the same
//!   event out again.
//! - **51** refuses a count at or past [`REQUEST_GROUPS`] with [`VI_OUT_OF_RANGE`] and a second 51
//!   with [`VI_PERMISSION_DENIED`]; otherwise it records the count. Nothing else reads it (PR:
//!   "Only the driver being up is required; `ConfigureRequestGroups` is not").
//! - **101** refuses a name other than [`DEFAULT_DISPLAY`] with `NotFound`, flags other than
//!   [`OPEN_DISPLAY_FLAGS`] with `NotImplemented`, and a display already opened
//!   [`MAX_DISPLAY_REFERENCES`] times with `OutOfResource`; otherwise it counts one more
//!   reference. The display is open while the count is not zero.
//! - **102** on [`DEFAULT_DISPLAY`] refuses a display that is not open with `NotFound`;
//!   otherwise it counts one reference less, and when that was the last it raises the display's
//!   notification ([`OPEN_DISPLAY_FLAGS`], the flags PR keeps as `m_display_notification`). On a
//!   player display's name (`"player1"` to `"player3"`, PR `GetPlayerDisplayIndex`) it answers
//!   `NotFound`, PR's answer for a player display with no reference, which is every player
//!   display here (below, "Where this differs"); on any other name, `InvalidCombination`.
//! - **104** and **106** answer [`VI_NOT_FOUND`] unless the name is [`DEFAULT_DISPLAY`] and the
//!   display is open; then 104 answers [`HDR_FORMAT_SDR`] and 106 [`BUFFER_RESOLUTION_720P`].
//! - **105**, **107** and **354** answer [`VI_OUT_OF_RANGE`] for a zero name, [`VI_NOT_FOUND`] for a
//!   name not in [`KNOWN_DISPLAYS`] ([`is_known_display`]), and their constant otherwise, open or
//!   not (PR: "The reply is a field of the display's driver record, which reads as 0 (105) and 1
//!   (354) while no record exists"; 107 "with no record they read as 1, 1 and 0").
//! - **111** refuses, with `NotFound`, a zero client, a client not registered, one not bound to
//!   a display, a display not open, a name other than [`DEFAULT_DISPLAY`] and a display name
//!   other than the name; then an endpoint already open with `InvalidState`. Otherwise the
//!   client is the endpoint. **112** refuses a name other than [`DEFAULT_DISPLAY`] or a display
//!   name other than it with `InvalidCombination`, and no endpoint with `NotFound`; otherwise it
//!   closes the endpoint.
//! - **201** refuses a zero client, a bit at or past [`CLIENT_NOTIFICATIONS`], or the same bit
//!   twice with `InvalidCombination`; a client already registered, or a bit another client holds,
//!   with `InvalidState`; and a session with [`MAX_CLIENTS`] clients with `OutOfResource`.
//!   Otherwise the client is registered with every slot free ([`ALL_SLOTS`]).
//! - **202** refuses a zero client with `InvalidCombination` and one not registered with
//!   `NotFound`; otherwise it closes the endpoint if the client holds it, destroys its layer,
//!   frees its bits, forgets it, and then raises its retire notification.
//! - **212** refuses a zero client with `InvalidCombination`, and one not registered or a display
//!   not open with `NotFound`; otherwise it makes the client's layer on first use and answers its
//!   free slots. A layer number past `u64::MAX` is `OutOfResource` (starquake's own).
//! - **251** refuses a zero client or one not registered with `NotFound`; on
//!   [`DEFAULT_DISPLAY`], a display not open with `NotFound` and a client already bound with
//!   `InvalidState`; on a player display's name, `InvalidState` (PR's answer for a player display
//!   with no reference); on any other name, `NotFound`. Otherwise it binds the client.
//! - **252** refuses a zero client, one not registered, or one not bound with `NotFound`;
//!   otherwise it unbinds it.
//! - **261–272**: the kind is `(command - 261) / 2` ([`monitor_kind`]). Registering refuses a
//!   group at or past [`REQUEST_GROUPS`] or a zero name with [`VI_OUT_OF_RANGE`], a name not in
//!   [`KNOWN_DISPLAYS`] or a group with no event with [`VI_NOT_FOUND`], and a group already
//!   registered with [`VI_PERMISSION_DENIED`]; otherwise it registers the group for the name
//!   and kind, and a kind other than [`VSYNC_KIND`] signals the group's event once (PR: "kinds
//!   1..5 ... also signal the group event once at registration"). Unregistering never fails
//!   (PR: "the real impls return nothing"): a registration that matches the group, the name and
//!   the kind is taken out and the group's event signalled; anything else changes nothing.
//! - **301** refuses a name other than [`DEFAULT_DISPLAY`] with `InvalidCombination`, and takes
//!   any value and mask on it. **conflict:** PR accepts only the mask [`DISPLAY_FLAG_63`] with a
//!   value of 0 or [`DISPLAY_FLAG_63`] (`IsSupportedDisplayParameter301`: "only those two forms
//!   are accepted", the two it saw the SDK send), though it reads the system module's handler as
//!   taking any (`(flags & ~mask) | (value & mask)`, "vi2.autogen 0xf37e0"); NX2 auto-stubs
//!   301 with success. The module's reading contains PR's two forms, so decision 8 takes it.
//!   Nothing keeps the flags: PR keeps nothing either, and nothing reads them yet.
//! - **351** refuses a zero client or one not registered with `NotFound`, and an encoding other
//!   than [`OUTPUT_ENCODING`] with `NotImplemented`. That is every change it could make, so it
//!   keeps nothing.
//! - **352** refuses a zero client or one not registered with `NotFound`, and levels whose first
//!   two are not within 0 to 1 or whose third is negative (a NaN is neither) with
//!   [`VI_OUT_OF_RANGE`] ([`levels_are_valid`]); otherwise it keeps them. **conflict:** PR's
//!   code answers `InvalidCombination` for those levels, though its own comment reads the system
//!   module's handler as answering VI result 4 ("vi2 0x63 handler 0xf4170 ... The first two must
//!   be within [0, 1] and the third not negative (otherwise VI result 4)"); NX2 has no 352 (it
//!   is auto-stubbed: success). Decision 8 takes the reading that cites the system module, so
//!   the refusal is [`VI_OUT_OF_RANGE`] (`output_levels_are_refused_with_the_module_s_result`).
//!   A title trace of a refused 352 would settle it.
//!
//! A request that lacks a data word, the handle or the buffer its command reads is the
//! framework's `INVALID_CMIF_REQUEST`, before anything changes; one with more data words is read
//! (PR compares the header words with literals and answers `InvalidCombination`; NX2 reads what
//! comes). A failure reply is the result word alone (the framework's rule): PR also writes its
//! data words after a failure (104's format, 107's three values, 212's mask), as zero.
//!
//! **Logs.** 104 to 107 and 354, which a title calls about 75 times a second (row 119), log at
//! trace level only; a refused one is still logged at warning by the framework, as every failed
//! command is. The vsync fallback is logged at info (below, "Vsync"), a clock with no room for a
//! group at error, and 201's refusal of a bit at or past [`CLIENT_NOTIFICATIONS`] and 351's of
//! an encoding at warning, each before the framework's own warning of the failed command.
//! `tests/vi2_log_levels.rs` pins each of these: the fallback, the full clock, the display
//! state commands, and 201's and 351's warnings (against a 201 and a 351 refused for another
//! reason, which log only the framework's).
//!
//! **Notifications.** A notification is a bit of a [`NOTIFICATION_WORDS`]-word bitmap. Raising
//! one sets its bit and signals the notification event, if 11 has made it. A signal the kernel
//! refuses is the request's error, after the change it notifies of is made.
//!
//! **Layers.** A registered client owns at most one layer (spec, "Layers"), made by its first
//! 212 on the open display and destroyed by 202. A layer is a number, counted up per session
//! from 1 ([`FIRST_LAYER`]), which nothing outside the session reads yet: slice 14 hands it to
//! the GPU with each present. Two sessions' layers may share a number; composition of several
//! layers is deferred (spec, "Deferred, not ruled out").
//!
//! # Vsync
//!
//! **conflict (spec, "Vsync events: conflict").** NX2 makes a new event on every 31 and 32 and
//! ticks every one at 60 Hz; PR reads 31 and 32 as a request group's event, ticked only once 261
//! registers the group for kind 0, the vsync. The spec follows PR and keeps NX2's behaviour as a
//! fallback: a group event with no registration is ticked anyway, and the fallback is logged.
//! Here:
//!
//! - **31** makes the event listen to the vsync clock at once ([`Locked::listen_vsync`], which
//!   starts a stopped clock one interval after the `Env` clock's reading), logged at info as the
//!   fallback.
//! - **261 (kind 0)** keeps it listening, as PR's `StartGroupTick` ticks it, and makes it listen
//!   again if a full clock had refused it.
//! - **263–271 (kinds 1 to 5)** take it out of the listeners ([`Locked::unlisten_vsync`]): the
//!   group has a registration, and not the vsync's, so neither PR's tick nor the fallback
//!   applies.
//! - **An unregister that matches** leaves the group with no registration: it listens again,
//!   logged at info as the fallback.
//!
//! So a group's event listens exactly while it has no registration or a kind-0 one. Where a
//! full clock ([`crate::shared::vsync::MAX_LISTENERS`]) refuses a group that should listen, 31
//! answers `OutOfResource` having made nothing, and 261 and an unregister log an error and
//! succeed, leaving the group not ticked. A group left registered for kind 0 listens again
//! once an unregister that matches finds room (`a_full_clock_leaves_a_registered_group_unticked`);
//! one left with no registration listens again once a 261 (kind 0) finds room
//! (`a_kind_zero_registration_puts_an_unticked_group_back_on_the_clock`).
//!
//! **The fallback's first frame.** The spec ticks an unregistered group "after the first frame".
//! The clock signals its listeners on every tick and offers vi2 no turn of its own, so the grace
//! of one frame is not built: an unregistered group is ticked from the clock's first tick on.
//! This departs from the spec's wording and is an owner question; a title that
//! registers kind 0 before the first tick, or waits on the event for two ticks as row 82's did,
//! sees no difference.
//!
//! The vsync clock's own rules (one clock for every session, no banking, the cap, refused
//! signals, what is recorded) are [`crate::shared::vsync`]'s.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `vi2/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | 1 and 3 | 0 | 1 | **conflict**: 1, the larger answer and the one a title ran on (decision 8; `the_limits_are_the_chosen_answers`). An owner question: see [`NOTIFICATION_LIMIT`] for the coupling with 201's bound, the row below |
//! | 1's answer and 201's bound together | 1 answers 0 (one word, bits 0 to 31), and 201 takes bits below 32 | 1 answers 1; 201 auto-stubbed (any bits) | 1 answers 1, two words (bits 0 to 63) if the SDK reads it as PR's comment says, while 201 and a restore take only bits below [`CLIENT_NOTIFICATIONS`], 32: a combination neither reference has, so a client the SDK gives a bit from 32 to 63 is refused (`a_client_bit_in_the_second_word_the_limit_names_is_refused`). Part of the owner question on 1 and 3 |
//! | 11 | one event per session | a new event per call | PR's (`the_notification_event_is_made_once_and_signalled_while_raised`) |
//! | 11 and 13 with a word other than 0 | `InvalidCombination` | 11 takes any word; 13 auto-stubbed | **conflict**: any word, NX2's, the one a title ran on (`notification_commands_take_any_word`) |
//! | 31 and 32 | a group's event, ticked once registered for kind 0 | a new event per call, every one ticked | **conflict**: PR's, with NX2's ticking as the fallback (above; `an_unregistered_group_is_ticked_by_the_fallback`, `a_kind_zero_monitor_ticks_its_group`, `a_kind_zero_registration_puts_an_unticked_group_back_on_the_clock`) |
//! | The handle kind of 11, 31 and 32 | moved | copied | **conflict**: copied, as [`crate::protocol::HANDLE_KIND`] (`the_request_group_event_is_handed_out_copied`; 11 and 32 in `every_command_s_reply_bytes`) |
//! | 32 with a handle naming a live object that is not an event, and a group out of range | `InvalidHandle` (the handle is checked for a readable event first) | auto-stubbed | [`VI_OUT_OF_RANGE`]: the group is checked before the identity, and the server cannot tell an event from another object (`duplicate_checks_the_group_before_the_identity`) |
//! | 104, 105 | as here | success, no data | PR's (`every_command_s_reply_bytes`) |
//! | Player displays (`"player1"`–`"player3"`) | opened, bound and closed for `GameShare` | auto-stubbed | not opened (101 `NotFound`); 102 and 251 answer as PR does for one with no reference (`player_displays_are_not_opened`). Deferred with other displays (spec) |
//! | 352 with levels it refuses | `InvalidCombination` (its code), VI result 4 (its comment, reading the system module) | auto-stubbed: success | **conflict**: [`VI_OUT_OF_RANGE`], the reading that cites the module (decision 8; `output_levels_are_refused_with_the_module_s_result`) |
//! | 301 with a mask or value other than PR's two forms | `InvalidCombination` | auto-stubbed: success | **conflict**: accepted on the default display, as PR reads the system module (`display_parameter_301_takes_any_flags_on_the_default_display`) |
//! | 106 | Eden's display resolution: 1 at 1280 × 720, `NotImplemented` otherwise | auto-stubbed | 1: the display is 1280 × 720 (handheld) until output sizes are read from a display description (spec, "Deferred") |
//! | A failure's data words | written as zero | — | not written: the result word alone (`a_failure_reply_is_the_result_word_alone`) |
//! | Header words other than PR's | `InvalidCombination` | read as they come | read as they come; only a missing word is refused (`a_short_request_is_refused_and_a_longer_one_is_read`) |
//! | A closed session | its layers destroyed, its events closed, its group tick stopped (destructor) | its events closed | nothing: the framework tells a session nothing when it closes. Its group events stay made, and those that listen keep listening, so they count against [`crate::shared::vsync::MAX_LISTENERS`]: after eight sessions that made eight listening groups each are closed, a new 31 answers `OutOfResource` (`closed_sessions_keep_their_groups_listening`). **Not bounded:** a group event registered for a kind other than 0 does not listen, so it does not count, and neither does a session's notification event: a title that keeps opening sessions, making such events and closing the sessions grows the kernel's events without limit. A known gap, left for a follow-up that needs a close hook from the framework |
//!
//! # What is recorded and what is saved
//!
//! Every request is one IPC event, recorded by the service framework with its payload. The only
//! input the session takes from outside the guest is the `Env` clock's reading when a group
//! starts listening, which the environment records; the ticks are the vsync clock's notes. A
//! session is plain data saved with the server ([`saved`]), checked as it is read back
//! ([`saved::Vi2SessionError`]). The family saves nothing ([`Vi2Family`]); the listeners are
//! saved with the vsync clock ([`crate::shared::Shared::save_vsync`]).

pub mod saved;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{
    DisplayId, Fence, ImageDesc, LayerId, PresentRequest, Rect, SyncpointId,
};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, MODULE_BITS, ResultCode};
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, Outcome, Reply, Request, Server, SnapshotError,
    UnknownPolicy,
};
use starquake_trace::HostNanos;

use crate::gpu::low_byte;
use crate::native::layout::{
    FORMAT_RGB10A2, FORMAT_RGBA8, WORD_BLOCK_HEIGHT, WORD_EXTENT, WORD_FORMAT, WORD_PITCH, extent,
};
use crate::shared::present::{NewFrame, QueueError, Refresh, RegisterError, slot_bit, write_slots};
use crate::shared::vsync::MAX_LISTENERS;
use crate::shared::{Locked, Shared};
use command::{
    BIND_CLIENT_DISPLAY, CANCEL_CLIENT_BUFFER, CLOSE_DISPLAY, CLOSE_ENDPOINT, CONFIGURE_REQUEST_GROUPS,
    CREATE_REQUEST_GROUP_EVENT, DUPLICATE_REQUEST_GROUP_EVENT, GET_BUFFER_MAX_RESOLUTION,
    GET_CLIENT_AVAILABILITY, GET_DISPLAY_STATE_105, GET_DISPLAY_STATE_107, GET_DISPLAY_STATE_354,
    GET_HDR_STATUS, GET_NOTIFICATION_EVENT, GET_NOTIFICATION_LIMIT, GET_NOTIFICATIONS,
    GET_REQUEST_GROUP_LIMIT, GET_SESSION_LIMIT, OPEN_DISPLAY, OPEN_ENDPOINT,
    PRESENT_CLIENT_BUFFER, REGISTER_BUFFER_MAX_RESOLUTION_MONITOR, REGISTER_CLIENT_STATE, REGISTER_DISPLAY_EVENT_MONITOR,
    REGISTER_DISPLAY_MONITOR_263, REGISTER_DISPLAY_MONITOR_267, REGISTER_DISPLAY_MONITOR_271,
    REGISTER_HDR_STATUS_MONITOR, RELEASE_NOTIFICATION_EVENT, SET_CLIENT_OUTPUT_ENCODING,
    SET_CLIENT_OUTPUT_LEVELS, SET_DISPLAY_PARAMETER_301, UNBIND_CLIENT_DISPLAY,
    UNREGISTER_BUFFER_MAX_RESOLUTION_MONITOR, UNREGISTER_CLIENT_STATE,
    UNREGISTER_DISPLAY_EVENT_MONITOR, UNREGISTER_DISPLAY_MONITOR_264,
    UNREGISTER_DISPLAY_MONITOR_268, UNREGISTER_DISPLAY_MONITOR_272, UNREGISTER_HDR_STATUS_MONITOR,
};

#[cfg(test)]
mod tests;

/// The name of the vi2 server. Starquake's own: the session is reached only through gsv 305, so
/// it has no registered name in either reference (NX2 logs it as `"nx2:vi2"`, PR as
/// `"IVi2Session"`).
pub const VI2_NAME: &str = "gsv:vi2";

/// The vi2 session's TIPC command ids, each with its source: PR's table
/// (`{id, &IVi2Session::..., "name"}`) and, where it has the id, NX2's.
pub mod command {
    /// PR `{1, &IVi2Session::GetSessionLimit, "GetNotificationLimit"}`; NX2 `"Unknown1"`.
    pub const GET_NOTIFICATION_LIMIT: u32 = 1;
    /// PR `{3, &IVi2Session::GetSessionLimit, "GetSessionLimit"}`; NX2 `"GetDisplayCount
    /// (guessed)"`.
    pub const GET_SESSION_LIMIT: u32 = 3;
    /// PR `{5, &IVi2Session::GetSessionLimit, "GetRequestGroupLimit"}`; NX2
    /// `"GetTimestampRecorderCount (guessed)"`.
    pub const GET_REQUEST_GROUP_LIMIT: u32 = 5;
    /// PR `{11, &IVi2Session::GetNotificationEvent, "GetNotificationEvent"}`; NX2 `"Unknown11"`.
    pub const GET_NOTIFICATION_EVENT: u32 = 11;
    /// PR `{13, &IVi2Session::ReleaseNotificationEvent, "ReleaseNotificationEvent"}`.
    pub const RELEASE_NOTIFICATION_EVENT: u32 = 13;
    /// PR `{21, &IVi2Session::GetNotifications, "GetNotifications"}`.
    pub const GET_NOTIFICATIONS: u32 = 21;
    /// PR `{31, &IVi2Session::CreateRequestGroupEvent, "CreateRequestGroupEvent"}`; NX2
    /// `"GetVsyncEvent (guessed)"`.
    pub const CREATE_REQUEST_GROUP_EVENT: u32 = 31;
    /// PR `{32, &IVi2Session::DuplicateRequestGroupEvent, "DuplicateRequestGroupEvent"}`; NX2
    /// `"GetDisplayVsyncEvent (guessed)"`.
    pub const DUPLICATE_REQUEST_GROUP_EVENT: u32 = 32;
    /// PR `{51, &IVi2Session::ConfigureRequestGroups, "ConfigureRequestGroups"}`.
    pub const CONFIGURE_REQUEST_GROUPS: u32 = 51;
    /// PR `{101, &IVi2Session::OpenDisplay, "OpenDisplay"}`.
    pub const OPEN_DISPLAY: u32 = 101;
    /// PR `{102, &IVi2Session::CloseDisplay, "CloseDisplay"}`.
    pub const CLOSE_DISPLAY: u32 = 102;
    /// PR `{104, &IVi2Session::GetHdrStatus, "GetHdrStatus"}`; NX2 `"Unknown104 (display
    /// \"default\")"`.
    pub const GET_HDR_STATUS: u32 = 104;
    /// PR `{105, &IVi2Session::GetDisplayState105, "GetDisplayState105"}`; NX2 `"Unknown105
    /// (display \"default\")"`.
    pub const GET_DISPLAY_STATE_105: u32 = 105;
    /// PR `{106, &IVi2Session::GetBufferMaxResolution, "GetBufferMaxResolution"}`.
    pub const GET_BUFFER_MAX_RESOLUTION: u32 = 106;
    /// PR `{107, &IVi2Session::GetDisplayState107, "GetDisplayState107"}`.
    pub const GET_DISPLAY_STATE_107: u32 = 107;
    /// PR `{111, &IVi2Session::OpenEndpoint, "OpenEndpoint"}`.
    pub const OPEN_ENDPOINT: u32 = 111;
    /// PR `{112, &IVi2Session::CloseEndpoint, "CloseEndpoint"}`.
    pub const CLOSE_ENDPOINT: u32 = 112;
    /// PR `{201, &IVi2Session::RegisterClientState, "RegisterClientState"}`.
    pub const REGISTER_CLIENT_STATE: u32 = 201;
    /// PR `{202, &IVi2Session::UnregisterClientState, "UnregisterClientState"}`.
    pub const UNREGISTER_CLIENT_STATE: u32 = 202;
    /// PR `{212, &IVi2Session::GetClientAvailability, "GetClientAvailability"}`.
    pub const GET_CLIENT_AVAILABILITY: u32 = 212;
    /// PR `{228, &IVi2Session::CancelClientBuffer, "CancelClientBuffer"}`; NX2 `{228,
    /// &INx2Vi2Session::CancelClientBuffer, "CancelClientBuffer"}`.
    pub const CANCEL_CLIENT_BUFFER: u32 = 228;
    /// PR `{229, &IVi2Session::PresentClientBuffer, "PresentClientBuffer"}`; NX2 `{229,
    /// &INx2Vi2Session::PresentClientBuffer, "PresentClientBuffer"}`.
    pub const PRESENT_CLIENT_BUFFER: u32 = 229;
    /// PR `{251, &IVi2Session::BindClientDisplay, "BindClientDisplay"}`.
    pub const BIND_CLIENT_DISPLAY: u32 = 251;
    /// PR `{252, &IVi2Session::UnbindClientDisplay, "UnbindClientDisplay"}`.
    pub const UNBIND_CLIENT_DISPLAY: u32 = 252;
    /// PR `{261, &IVi2Session::RegisterRequestGroupMonitor, "RegisterDisplayEventMonitor"}`:
    /// the first display monitor command (kind 0, the vsync).
    pub const REGISTER_DISPLAY_EVENT_MONITOR: u32 = 261;
    /// PR `{262, &IVi2Session::UnregisterRequestGroupMonitor, "UnregisterDisplayEventMonitor"}`.
    pub const UNREGISTER_DISPLAY_EVENT_MONITOR: u32 = 262;
    /// PR `{263, &IVi2Session::RegisterRequestGroupMonitor, "RegisterDisplayMonitor263"}`.
    pub const REGISTER_DISPLAY_MONITOR_263: u32 = 263;
    /// PR `{264, &IVi2Session::UnregisterRequestGroupMonitor, "UnregisterDisplayMonitor264"}`.
    pub const UNREGISTER_DISPLAY_MONITOR_264: u32 = 264;
    /// PR `{265, &IVi2Session::RegisterRequestGroupMonitor, "RegisterHdrStatusMonitor"}`.
    pub const REGISTER_HDR_STATUS_MONITOR: u32 = 265;
    /// PR `{266, &IVi2Session::UnregisterRequestGroupMonitor, "UnregisterHdrStatusMonitor"}`.
    pub const UNREGISTER_HDR_STATUS_MONITOR: u32 = 266;
    /// PR `{267, &IVi2Session::RegisterRequestGroupMonitor, "RegisterDisplayMonitor267"}`.
    pub const REGISTER_DISPLAY_MONITOR_267: u32 = 267;
    /// PR `{268, &IVi2Session::UnregisterRequestGroupMonitor, "UnregisterDisplayMonitor268"}`.
    pub const UNREGISTER_DISPLAY_MONITOR_268: u32 = 268;
    /// PR `{269, &IVi2Session::RegisterRequestGroupMonitor,
    /// "RegisterBufferMaxResolutionMonitor"}`.
    pub const REGISTER_BUFFER_MAX_RESOLUTION_MONITOR: u32 = 269;
    /// PR `{270, &IVi2Session::UnregisterRequestGroupMonitor,
    /// "UnregisterBufferMaxResolutionMonitor"}`.
    pub const UNREGISTER_BUFFER_MAX_RESOLUTION_MONITOR: u32 = 270;
    /// PR `{271, &IVi2Session::RegisterRequestGroupMonitor, "RegisterDisplayMonitor271"}`.
    pub const REGISTER_DISPLAY_MONITOR_271: u32 = 271;
    /// PR `{272, &IVi2Session::UnregisterRequestGroupMonitor, "UnregisterDisplayMonitor272"}`:
    /// the last display monitor command.
    pub const UNREGISTER_DISPLAY_MONITOR_272: u32 = 272;
    /// The display monitor commands, in order: PR's rows 261 to 272, registering on the odd ids
    /// and unregistering on the even ones.
    pub const MONITORS: [u32; 12] = [
        REGISTER_DISPLAY_EVENT_MONITOR,
        UNREGISTER_DISPLAY_EVENT_MONITOR,
        REGISTER_DISPLAY_MONITOR_263,
        UNREGISTER_DISPLAY_MONITOR_264,
        REGISTER_HDR_STATUS_MONITOR,
        UNREGISTER_HDR_STATUS_MONITOR,
        REGISTER_DISPLAY_MONITOR_267,
        UNREGISTER_DISPLAY_MONITOR_268,
        REGISTER_BUFFER_MAX_RESOLUTION_MONITOR,
        UNREGISTER_BUFFER_MAX_RESOLUTION_MONITOR,
        REGISTER_DISPLAY_MONITOR_271,
        UNREGISTER_DISPLAY_MONITOR_272,
    ];
    /// PR `{301, &IVi2Session::SetDisplayParameter301, "SetDisplayParameter301"}`.
    pub const SET_DISPLAY_PARAMETER_301: u32 = 301;
    /// PR `{351, &IVi2Session::SetClientOutputEncoding, "SetClientOutputEncoding"}`.
    pub const SET_CLIENT_OUTPUT_ENCODING: u32 = 351;
    /// PR `{352, &IVi2Session::SetClientOutputLevels, "SetClientOutputLevels"}`.
    pub const SET_CLIENT_OUTPUT_LEVELS: u32 = 352;
    /// PR `{354, &IVi2Session::GetDisplayState354, "GetDisplayState354"}`.
    pub const GET_DISPLAY_STATE_354: u32 = 354;
}

/// `ErrorModule::VI` (`eden/src/core/hle/result.h` @ cf654052).
pub const VI_MODULE: u32 = 114;

/// A VI result with `description`.
const fn vi(description: u32) -> ResultCode {
    ResultCode::from_raw(VI_MODULE | (description << MODULE_BITS))
}

/// PR's `Result{ErrorModule::VI, 4}`: a group index out of range, a zero display name, a 51
/// count out of range; and 352's refused levels, which PR's comment reads from the system module
/// ("otherwise VI result 4") though its code answers `InvalidCombination` (module docs,
/// "Commands": a **conflict**). The name is starquake's; no reference names description 4.
pub const VI_OUT_OF_RANGE: ResultCode = vi(4);
/// PR's `Result{ErrorModule::VI, 5}`: a group's event or registration that exists, a second 51.
/// Eden's `ResultPermissionDenied{ErrorModule::VI, 5}` (`vi_results.h`).
pub const VI_PERMISSION_DENIED: ResultCode = vi(5);
/// PR's `Result{ErrorModule::VI, 7}`: a display name not known or not open, a group with no
/// event. Eden's `ResultNotFound{ErrorModule::VI, 7}` (`vi_results.h`).
pub const VI_NOT_FOUND: ResultCode = vi(7);

/// What 1 answers. A **conflict**: PR's `GetSessionLimit` answers 0 for every id but 5, NX2's
/// `ReturnOne` 1 ("TIPC 1 -> u32 < 0x400"). Neither cites the system module for the value; 1
/// contains 0 as an inclusive upper index (PR: "The SDK converts this inclusive upper group
/// index into 32 client slots") and is the one a title ran on, so decision 8 takes it.
///
/// **Coupled with 201's bound, and an owner question.** Read as PR's comment reads it, 1 is an
/// inclusive upper word index, so the SDK may hand clients bits from 0 to 63
/// (`(NOTIFICATION_LIMIT + 1) * 32`). 201 and a restore still take only bits below
/// [`CLIENT_NOTIFICATIONS`] (32, PR's `first < 32 && second < 32`, which matches PR's answer of
/// 0), and the saved session refuses a bit raised past word 0. Answering 1 with 201 bounded at
/// 32 is a combination neither reference has: a client given a bit from 32 to 63 is refused with
/// `InvalidCombination` (`a_client_bit_in_the_second_word_the_limit_names_is_refused`). Whether
/// to answer 0, or to widen 201, the client masks and the saved check to
/// `(NOTIFICATION_LIMIT + 1) * 32`, is left to the owner with the 1 and 3 conflict (module docs,
/// "Where this differs").
pub const NOTIFICATION_LIMIT: u32 = 1;
/// What 3 answers: the same **conflict** as [`NOTIFICATION_LIMIT`] (NX2 "TIPC 3 -> u32 < 8").
pub const SESSION_LIMIT: u32 = 1;
/// What 5 answers: PR's `GetSessionLimit` (`ctx.GetCommand() == 5 ? 7 : 0`) and NX2's
/// `GetTimestampRecorderCount` agree on 7 (row 21: "`LayerHistory` creates 7 timestamp
/// recorders"), the last index of [`REQUEST_GROUPS`] groups.
pub const REQUEST_GROUP_LIMIT: u32 = 7;

/// The request groups a session has: PR `RequestGroupCount = 8`
/// (`graphics_supervisor_protocol.h`), its `m_request_group_events` and
/// `m_request_group_displays` arrays.
pub const REQUEST_GROUPS: u32 = 8;
/// The words of the notification bitmap: PR `std::array<u32, 32> m_notifications`.
pub const NOTIFICATION_WORDS: usize = 32;
/// The bytes 21 writes, and the least its out buffer may hold: PR
/// `ctx.GetWriteBufferSize() >= sizeof(m_notifications)`.
pub const NOTIFICATIONS_SIZE: u64 = 4 * NOTIFICATION_WORDS as u64;
/// The notification bits a client may name: PR `RegisterClientState`, `first < 32 && second <
/// 32`, which its `u32 slots` mask holds.
pub const CLIENT_NOTIFICATIONS: u32 = 32;
/// The clients a session holds: PR `std::array<ClientState, 16> m_client_states`.
pub const MAX_CLIENTS: usize = 16;
/// The references 101 counts at most: PR `R_UNLESS(m_display_references < 1024,
/// Kernel::ResultOutOfResource)`.
pub const MAX_DISPLAY_REFERENCES: u32 = 1024;
/// The default display's name, `"default"` as eight bytes: PR `0x00746c7561666564ULL`.
pub const DEFAULT_DISPLAY: u64 = u64::from_le_bytes(*b"default\0");
/// The flags 101 accepts, which PR keeps as the display's notification
/// (`R_UNLESS(flags == 1, Kernel::ResultNotImplemented)`, `m_display_notification = flags`).
pub const OPEN_DISPLAY_FLAGS: u32 = 1;
/// The buffer slots a client has: PR `slot < 15` (228, 229) and `std::array<std::optional<s32>,
/// 15> queued`; spec, "Fifteen slots".
pub const SLOTS: u8 = 15;
/// A fresh client's free slots: fifteen, PR `u64 available{0x7fff}`; NX2's 229 reply
/// `AvailableSlots = 0x7fff` (row 48).
pub const ALL_SLOTS: u64 = 0x7FFF;
/// The layout 229 accepts at word 8: PR `packet[8] == 3`; spec "u32 layout (w8, 3)".
pub const PRESENT_LAYOUT: u32 = 3;
/// The block height (log2 of GOB rows, preference word 4) a presented image must have: PR
/// `image->preference[4] == 7` ("the backend's 16-GOB block height" in its comment).
pub const PRESENT_BLOCK_HEIGHT: u32 = 7;
/// The formats (preference word 6) a presented image may have: PR `image->preference[6] == 1 ||
/// image->preference[6] == 4` (spec: 1 RGBA8, 4 RGB10A2), the two 429 already takes.
pub const PRESENT_FORMATS: [u32; 2] = [FORMAT_RGBA8, FORMAT_RGB10A2];
/// The 229 request word that holds the layout (PR `packet[8]`), the last word before the fence
/// that 229 reads.
pub const PRESENT_WORD_LAYOUT: usize = 8;
/// The 229 request word that holds the fence's `SyncCounterId` (PR `~packet[22]`, NX2 `~w[22]`;
/// spec, "fence `SyncCounterId` (w22) and value (w24)"); words 23 and 25 must be 0 (PR
/// `packet[23] == 0 && packet[25] == 0`), word 24 is the value.
pub const PRESENT_WORD_FENCE: usize = 22;
/// What 229's fence word is when the present has no fence: PR `fence_id == ~u32{0}` on `fence_id
/// = ~packet[22]`, so the word itself is 0.
pub const NO_FENCE: u32 = 0;

/// How 229 paces the title (spec, "Present (229) and pacing"; the owner's decision 1): a per-run
/// setting, given to the server ([`vi2_server_paced`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pacing {
    /// The buffer queue (decided by the owner, decision 1): 229 replies at once with the slots
    /// still free, and a slot comes back on the tick that latches the frame replacing it.
    #[default]
    BufferQueue,
    /// The fence wait fallback (NX2's): 229 holds its reply until the frame before it is
    /// latched and reports every slot free ([`crate::shared::present`], "The fence wait
    /// fallback").
    FenceWait,
}
/// The only output encoding 351 accepts, and a client's: PR `u64 output_encoding{0x14}`, "Only
/// the SDK's observed default output encoding is supported".
pub const OUTPUT_ENCODING: u64 = 0x14;
/// What 104 answers for the open default display: PR "The display backend exposes SDR output
/// only", `format = 1`.
pub const HDR_FORMAT_SDR: u64 = 1;
/// What 106 answers for the open default display: PR `resolution = 1` for a 1280 × 720 display.
pub const BUFFER_RESOLUTION_720P: u64 = 1;
/// What 105 answers for a known name: PR `ReplyDisplayState(ctx, 0x79, 3, 0)`.
pub const DISPLAY_STATE_105: u32 = 0;
/// What 107 answers for a known name: PR `rb.Push<u64>(ok ? 1 : 0)` twice, then `0`.
pub const DISPLAY_STATE_107: [u64; 3] = [1, 1, 0];
/// What 354 answers for a known name: PR `ReplyDisplayState(ctx, 0x172, 4, 1)`.
pub const DISPLAY_STATE_354: u64 = 1;
/// The display flag PR's 301 accepts alone (module docs, "Commands": a **conflict**): PR
/// `IsSupportedDisplayParameter301`, `Bit63 = 1ULL << 63`.
pub const DISPLAY_FLAG_63: u64 = 1 << 63;
/// The display monitor kind that is the vsync: PR "Kind 0 is a display native event ... It is
/// the vsync event".
pub const VSYNC_KIND: u32 = 0;
/// The display monitor kinds: 261 to 272 are six register and unregister pairs.
pub const MONITOR_KINDS: u32 = 6;
/// The first layer number of a session (module docs, "Layers"). Starquake's own.
pub const FIRST_LAYER: u64 = 1;

/// The names of the vi2 server's display table that PR's `IsKnownDisplayName` lists: "the real
/// vi2 server's 40-entry display table" (its 39-entry array, then `"#ld01"`, `"#ldv"` and
/// `"#ovl1"`, which it compares one by one).
pub const KNOWN_DISPLAYS: [&str; 42] = [
    "default", "cpyrght", "full", "overlay", "overlay0", "overlay1", "overlay2", "test",
    "adraw:s0", "adraw:s1", "adraw:s2", "adraw:c0", "adraw:c1", "adraw:i0", "player1", "player2",
    "player3", "chat0", "chat1", "chat2", "chat3", "chat4", "chat5", "chat6", "chat7", "chat8",
    "chat9", "chat10", "chat11", "chat12", "chat13", "chat14", "chat15", "chvcam", "null", "#ld",
    "#ld0", "#ld1", "#ovl0", "#ld01", "#ldv", "#ovl1",
];

/// Whether `name` names a display of [`KNOWN_DISPLAYS`], as PR's `IsKnownDisplayName` reads it:
/// the eight bytes as text up to the first zero byte (`std::string_view{text}` over a
/// zero-terminated copy), so bytes after a zero byte are not looked at.
#[must_use]
pub fn is_known_display(name: u64) -> bool {
    let bytes = name.to_le_bytes();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = bytes.get(..end).unwrap_or_default();
    KNOWN_DISPLAYS.iter().any(|known| known.as_bytes() == text)
}

/// The player display a name opens, `"player1"` to `"player3"`: PR `GetPlayerDisplayIndex`
/// (`(name & 0xffffffffffffULL) != 0x726579616c70ULL || (name >> 56) != 0`, then the seventh
/// byte from `'1'` to `'3'`).
#[must_use]
pub fn player_display(name: u64) -> Option<u8> {
    let bytes = name.to_le_bytes();
    let (prefix, rest) = bytes.split_at(6);
    match (prefix, rest) {
        (b"player", &[digit @ b'1'..=b'3', 0]) => Some(digit - b'1'),
        _ => None,
    }
}

/// The display monitor kind of a monitor command and whether it registers: PR `(command - 261)
/// / 2` and `(command - 262) / 2`, the odd ids registering. `None` for an id outside 261 to 272.
#[must_use]
pub fn monitor_kind(command: u32) -> Option<(u32, bool)> {
    let offset = command.checked_sub(REGISTER_DISPLAY_EVENT_MONITOR)?;
    let kind = offset / 2;
    (kind < MONITOR_KINDS).then_some((kind, offset % 2 == 0))
}

/// Whether 352's levels are accepted, as PR checks them: each of the first two at least `0.0f`
/// and at most `1.0f`, the third at least `0.0f` (`levels[0] >= 0.0f && levels[0] <= 1.0f`, the
/// same for `levels[1]`, then `levels[2] >= 0.0f`), so a NaN anywhere is refused.
#[must_use]
pub fn levels_are_valid(levels: [f32; 3]) -> bool {
    let [first, second, third] = levels;
    (0.0..=1.0).contains(&first) && (0.0..=1.0).contains(&second) && third >= 0.0
}

/// A display monitor a request group is registered as (PR `DisplayMonitor`): the display's
/// name and the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Monitor {
    display: u64,
    kind: u32,
}

impl Monitor {
    /// The display's name.
    #[must_use]
    pub const fn display(&self) -> u64 {
        self.display
    }

    /// The kind: [`VSYNC_KIND`] or one of the others below [`MONITOR_KINDS`].
    #[must_use]
    pub const fn kind(&self) -> u32 {
        self.kind
    }
}

/// One request group: its event, once 31 made it, and its registration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestGroup {
    event: Option<ObjectId>,
    monitor: Option<Monitor>,
}

impl RequestGroup {
    /// The group's event, once 31 made it.
    #[must_use]
    pub const fn event(&self) -> Option<ObjectId> {
        self.event
    }

    /// The display monitor the group is registered as.
    #[must_use]
    pub const fn monitor(&self) -> Option<Monitor> {
        self.monitor
    }

    /// Whether the group's event listens to the vsync clock (module docs, "Vsync"): it has an
    /// event, and no registration or a kind-0 one.
    #[must_use]
    pub fn listens(&self) -> bool {
        self.event.is_some() && self.monitor.is_none_or(|m| m.kind == VSYNC_KIND)
    }
}

/// Notification `index`'s bit in a client's `u32` mask; none for an index at or past
/// [`CLIENT_NOTIFICATIONS`], which 201 and a restore refuse.
const fn bit(index: u32) -> u32 {
    match 1_u32.checked_shl(index) {
        Some(bit) => bit,
        None => 0,
    }
}

/// One registered client (PR `ClientState`, but its present queue, which is in gsv's shared
/// state: [`crate::shared::present`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Client {
    token: u64,
    retire: u32,
    available_bit: u32,
    bound: bool,
    layer: Option<u64>,
    available: u64,
    levels: Option<[u32; 3]>,
}

impl Client {
    /// A client 201 registers: no display, no layer, every slot free.
    const fn new(token: u64, retire: u32, available_bit: u32) -> Self {
        Self {
            token,
            retire,
            available_bit,
            bound: false,
            layer: None,
            available: ALL_SLOTS,
            levels: None,
        }
    }

    /// The client's token.
    #[must_use]
    pub const fn token(&self) -> u64 {
        self.token
    }

    /// The notification 202 raises: PR `retire_notification`.
    #[must_use]
    pub const fn retire(&self) -> u32 {
        self.retire
    }

    /// The notification a freed slot raises: PR `available_notification`.
    #[must_use]
    pub const fn available_bit(&self) -> u32 {
        self.available_bit
    }

    /// The two notification bits the client holds, as a mask.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        bit(self.retire) | bit(self.available_bit)
    }

    /// Whether 251 bound it to the default display.
    #[must_use]
    pub const fn is_bound(&self) -> bool {
        self.bound
    }

    /// Its layer, once 212 made it.
    #[must_use]
    pub fn layer(&self) -> Option<LayerId> {
        self.layer.map(LayerId)
    }

    /// Its free slots.
    #[must_use]
    pub const fn available(&self) -> u64 {
        self.available
    }

    /// The levels 352 set, if it did.
    #[must_use]
    pub fn levels(&self) -> Option<[f32; 3]> {
        self.levels.map(|bits| bits.map(f32::from_bits))
    }
}

/// One vi2 session (module docs, "The server"). Deserialising goes through
/// [`saved::SavedVi2Session`], which refuses a state no sequence of commands leads to
/// ([`saved::Vi2SessionError`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "saved::SavedVi2Session")]
pub struct Vi2Session {
    notification_event: Option<ObjectId>,
    notifications: [u32; NOTIFICATION_WORDS],
    groups: Vec<RequestGroup>,
    group_count: Option<u32>,
    display_references: u32,
    clients: Vec<Client>,
    endpoint: Option<u64>,
    next_layer: u64,
}

impl Default for Vi2Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Vi2Session {
    /// A session no command has reached: no events, no notification raised, no group
    /// configured, the display not open, no client, no endpoint (PR's members, all zero or
    /// empty), and layers numbered from [`FIRST_LAYER`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            notification_event: None,
            notifications: [0; NOTIFICATION_WORDS],
            groups: vec![RequestGroup::default(); REQUEST_GROUPS as usize],
            group_count: None,
            display_references: 0,
            clients: Vec::new(),
            endpoint: None,
            next_layer: FIRST_LAYER,
        }
    }

    /// The notification event, once 11 made it.
    #[must_use]
    pub const fn notification_event(&self) -> Option<ObjectId> {
        self.notification_event
    }

    /// The notification bitmap.
    #[must_use]
    pub const fn notifications(&self) -> &[u32; NOTIFICATION_WORDS] {
        &self.notifications
    }

    /// The request groups, by index.
    #[must_use]
    pub fn groups(&self) -> &[RequestGroup] {
        &self.groups
    }

    /// The count 51 recorded.
    #[must_use]
    pub const fn group_count(&self) -> Option<u32> {
        self.group_count
    }

    /// How many times the default display is open.
    #[must_use]
    pub const fn display_references(&self) -> u32 {
        self.display_references
    }

    /// The registered clients, in the order 201 registered them.
    #[must_use]
    pub fn clients(&self) -> &[Client] {
        &self.clients
    }

    /// The client whose endpoint 111 opened.
    #[must_use]
    pub const fn endpoint(&self) -> Option<u64> {
        self.endpoint
    }

    /// The number the next layer gets.
    #[must_use]
    pub const fn next_layer(&self) -> u64 {
        self.next_layer
    }

    /// Whether the default display is open.
    const fn is_open(&self) -> bool {
        self.display_references != 0
    }

    /// The index of the client `token` names; never one for token 0, which no client has.
    fn client(&self, token: u64) -> Option<usize> {
        self.clients.iter().position(|c| c.token == token)
    }

    /// The registered client `token` names, refusing a zero or unknown one with `NotFound`.
    fn known_client(&mut self, token: u64) -> Result<&mut Client, Fault> {
        let index = self.client(token).ok_or(KernelError::NotFound)?;
        self.clients
            .get_mut(index)
            .ok_or_else(|| KernelError::NotFound.into())
    }

    /// The group `group` names, refusing one at or past [`REQUEST_GROUPS`] with
    /// [`VI_OUT_OF_RANGE`].
    fn group(&self, group: u32) -> Result<usize, Fault> {
        usize::try_from(group)
            .ok()
            .filter(|&index| index < self.groups.len())
            .ok_or_else(|| VI_OUT_OF_RANGE.into())
    }

    /// Takes into each client's free slots and the notification bitmap what the vsync ticks
    /// released for it since the last time (PR `RefreshReleasedBuffers`; module docs, "Present").
    fn refresh(&mut self, state: &mut Locked<'_>) {
        for client in &mut self.clients {
            let Refresh { released, raised } = state.presents_mut().take_refresh(client.token);
            client.available |= released & ALL_SLOTS;
            if raised {
                self.notifications[0] |= bit(client.available_bit);
            }
        }
    }

    /// Raises notification `index` and signals the notification event if 11 made it (module
    /// docs, "Notifications"). `index` is below [`CLIENT_NOTIFICATIONS`] or is
    /// [`OPEN_DISPLAY_FLAGS`], so it names a bit of the bitmap.
    fn notify(&mut self, cx: &mut Context<'_>, index: u32) -> Result<(), Fault> {
        let word = usize::try_from(index / 32).map_err(|_| KernelError::InvalidState)?;
        let bit = self
            .notifications
            .get_mut(word)
            .ok_or(KernelError::InvalidState)?;
        *bit |= 1 << (index % 32);
        if let Some(event) = self.notification_event {
            cx.signal_event(event)?;
        }
        Ok(())
    }
}

/// What every vi2 session shares: the handle to gsv's shared state, whose vsync clock ticks the
/// request groups. It is the run's wiring, not guest state, so a snapshot holds nothing of it,
/// as [`crate::syncpt::SyncptFamily`] holds nothing.
pub struct Vi2Family {
    shared: Arc<Shared>,
    pacing: Pacing,
}

impl Vi2Family {
    /// The family over gsv's `shared` state, pacing with the buffer queue.
    #[must_use]
    pub fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            pacing: Pacing::BufferQueue,
        }
    }

    /// The same family, pacing with `pacing`.
    #[must_use]
    pub fn with_pacing(self, pacing: Pacing) -> Self {
        Self { pacing, ..self }
    }

    /// gsv's shared state.
    #[must_use]
    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// How 229 paces the title.
    #[must_use]
    pub const fn pacing(&self) -> Pacing {
        self.pacing
    }
}

impl Family for Vi2Family {
    /// No bytes: the family holds only the run's wiring.
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    /// Accepts only what [`save_state`](Self::save_state) writes, and keeps the handle.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::Inconsistent("the vi2 family saves no state"))
        }
    }
}

/// gsv's shared state, from the server's family.
fn shared(cx: &mut Context<'_>) -> Result<Arc<Shared>, Fault> {
    Ok(Arc::clone(cx.family::<Vi2Family>()?.shared()))
}

/// How the server's family paces 229.
fn pacing(cx: &mut Context<'_>) -> Result<Pacing, Fault> {
    Ok(cx.family::<Vi2Family>()?.pacing())
}

/// The `Env` clock's reading on the handler's thread, which starts a stopped vsync clock.
fn now(cx: &Context<'_>) -> Result<HostNanos, Fault> {
    Ok(cx.env()?.now()?)
}

/// Makes `event`, request group `group`'s, listen to the vsync clock again at `now`, because it
/// has no registration or a kind-0 one (module docs, "Vsync"); logs a refusal as an error.
fn listen(state: &mut Locked<'_>, event: ObjectId, group: u32, now: HostNanos, fallback: bool) {
    match state.listen_vsync(event, now) {
        Ok(added) => {
            if fallback && added {
                tracing::info!(
                    group,
                    "gsv:vi2: a request group with no display monitor; the vsync clock ticks it anyway (fallback)"
                );
            }
        }
        Err(error) => {
            tracing::error!(
                %error,
                group,
                "gsv:vi2: the vsync clock refused a request group; it is not ticked"
            );
        }
    }
}

/// The words of a 229 request that 229 reads (module docs, "Present"), in request order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresentWords {
    /// w2–3: the client.
    token: u64,
    /// The low byte of w4: the buffer slot (PR `DecodeBufferSlot`).
    slot: u8,
    /// w5: the native image.
    image: u32,
    /// w6: the byte offset into the image.
    offset: u32,
    /// w7: the shown extent, `(width - 1) | (height - 1) << 16`.
    shown: u32,
    /// w8: the layout.
    layout: u32,
    /// w22: the fence's `SyncCounterId`, [`NO_FENCE`] for none.
    fence_id: u32,
    /// w23: 0.
    fence_high: u32,
    /// w24: the fence's value.
    fence_value: u32,
    /// w25: 0.
    fence_tail: u32,
}

impl PresentWords {
    /// Reads them from `req`; a request that ends before word 25 is the framework's
    /// `INVALID_CMIF_REQUEST`.
    fn read(req: &Request<'_>) -> Result<Self, Fault> {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let slot = low_byte(payload.read_u32()?);
        let image = payload.read_u32()?;
        let offset = payload.read_u32()?;
        let shown = payload.read_u32()?;
        let layout = payload.read_u32()?;
        for _ in PRESENT_WORD_LAYOUT + 1..PRESENT_WORD_FENCE {
            payload.read_u32()?;
        }
        Ok(Self {
            token,
            slot,
            image,
            offset,
            shown,
            layout,
            fence_id: payload.read_u32()?,
            fence_high: payload.read_u32()?,
            fence_value: payload.read_u32()?,
            fence_tail: payload.read_u32()?,
        })
    }
}

commands! {
    impl Vi2Session {}
    tipc {
        GET_NOTIFICATION_LIMIT => get_notification_limit,
        GET_SESSION_LIMIT => get_session_limit,
        GET_REQUEST_GROUP_LIMIT => get_request_group_limit,
        GET_NOTIFICATION_EVENT => get_notification_event,
        RELEASE_NOTIFICATION_EVENT => release_notification_event,
        GET_NOTIFICATIONS => get_notifications [Out(MapAlias)],
        CREATE_REQUEST_GROUP_EVENT => create_request_group_event,
        DUPLICATE_REQUEST_GROUP_EVENT => duplicate_request_group_event,
        CONFIGURE_REQUEST_GROUPS => configure_request_groups,
        OPEN_DISPLAY => open_display,
        CLOSE_DISPLAY => close_display,
        GET_HDR_STATUS => get_hdr_status,
        GET_DISPLAY_STATE_105 => get_display_state_105,
        GET_BUFFER_MAX_RESOLUTION => get_buffer_max_resolution,
        GET_DISPLAY_STATE_107 => get_display_state_107,
        OPEN_ENDPOINT => open_endpoint,
        CLOSE_ENDPOINT => close_endpoint,
        REGISTER_CLIENT_STATE => register_client_state,
        UNREGISTER_CLIENT_STATE => unregister_client_state,
        GET_CLIENT_AVAILABILITY => get_client_availability,
        CANCEL_CLIENT_BUFFER => cancel_client_buffer,
        PRESENT_CLIENT_BUFFER => present_client_buffer,
        BIND_CLIENT_DISPLAY => bind_client_display,
        UNBIND_CLIENT_DISPLAY => unbind_client_display,
        REGISTER_DISPLAY_EVENT_MONITOR => monitor,
        UNREGISTER_DISPLAY_EVENT_MONITOR => monitor,
        REGISTER_DISPLAY_MONITOR_263 => monitor,
        UNREGISTER_DISPLAY_MONITOR_264 => monitor,
        REGISTER_HDR_STATUS_MONITOR => monitor,
        UNREGISTER_HDR_STATUS_MONITOR => monitor,
        REGISTER_DISPLAY_MONITOR_267 => monitor,
        UNREGISTER_DISPLAY_MONITOR_268 => monitor,
        REGISTER_BUFFER_MAX_RESOLUTION_MONITOR => monitor,
        UNREGISTER_BUFFER_MAX_RESOLUTION_MONITOR => monitor,
        REGISTER_DISPLAY_MONITOR_271 => monitor,
        UNREGISTER_DISPLAY_MONITOR_272 => monitor,
        SET_DISPLAY_PARAMETER_301 => set_display_parameter_301,
        SET_CLIENT_OUTPUT_ENCODING => set_client_output_encoding,
        SET_CLIENT_OUTPUT_LEVELS => set_client_output_levels,
        GET_DISPLAY_STATE_354 => get_display_state_354,
    }
}

impl Vi2Session {
    /// `GetNotificationLimit` (1). out: `u32` [`NOTIFICATION_LIMIT`].
    fn get_notification_limit(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(NOTIFICATION_LIMIT);
        reply.ok()
    }

    /// `GetSessionLimit` (3). out: `u32` [`SESSION_LIMIT`].
    fn get_session_limit(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(SESSION_LIMIT);
        reply.ok()
    }

    /// `GetRequestGroupLimit` (5). out: `u32` [`REQUEST_GROUP_LIMIT`].
    fn get_request_group_limit(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(REQUEST_GROUP_LIMIT);
        reply.ok()
    }

    /// `GetNotificationEvent` (11). in: `u32`, read and logged, any value (module docs,
    /// "Commands"). out: copy handle, the notification event.
    fn get_notification_event(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let word = req.payload().read_u32()?;
        tracing::debug!(word, "gsv:vi2: notification event");
        let shared = shared(cx)?;
        let mut state = shared.lock();
        let event = if let Some(event) = self.notification_event {
            event
        } else {
            let event = cx.create_event()?;
            self.notification_event = Some(event);
            for client in &self.clients {
                state.presents_mut().set_event(client.token, event);
            }
            event
        };
        self.refresh(&mut state);
        if self.notifications.iter().any(|&word| word != 0) {
            cx.signal_event(event)?;
        }
        drop(state);
        reply.copy_handle(event);
        reply.ok()
    }

    /// `ReleaseNotificationEvent` (13). in: `u32`, read and logged, any value (module docs,
    /// "Commands").
    fn release_notification_event(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let word = req.payload().read_u32()?;
        tracing::debug!(word, "gsv:vi2: release notification event");
        reply.ok()
    }

    /// `GetNotifications` (21). out buffer 0 (a B descriptor; the kind is a **guess**: PR
    /// writes it with `ctx.WriteBuffer`, which takes either kind): the bitmap. Reading it
    /// clears it.
    fn get_notifications(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let shared = shared(cx)?;
        let mut out = req.out_buffer(cx, 0)?;
        if out.len() < NOTIFICATIONS_SIZE {
            return Err(KernelError::InvalidSize.into());
        }
        // Held to the clear, so that a tick's release between the refresh and the clear is not
        // wiped out by it (module docs, "Present").
        let mut state = shared.lock();
        self.refresh(&mut state);
        let bytes: Vec<u8> = self
            .notifications
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        out.write(&bytes)?;
        if let Some(event) = self.notification_event {
            cx.clear_event(event)?;
        }
        tracing::debug!(notifications = ?self.notifications, "gsv:vi2: notifications read");
        self.notifications = [0; NOTIFICATION_WORDS];
        drop(state);
        reply.ok()
    }

    /// `CreateRequestGroupEvent` (31). in: `u32` group. out: copy handle, the group's event,
    /// which starts listening to the vsync clock (module docs, "Vsync").
    fn create_request_group_event(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let group = req.payload().read_u32()?;
        let index = self.group(group)?;
        if self.groups.get(index).and_then(|g| g.event).is_some() {
            return Err(VI_PERMISSION_DENIED.into());
        }
        let now = now(cx)?;
        let shared = shared(cx)?;
        let mut state = shared.lock();
        if state.vsync().listeners().count() >= MAX_LISTENERS {
            tracing::error!(
                group,
                "gsv:vi2: the vsync clock has no room for a request group"
            );
            return Err(KernelError::OutOfResource.into());
        }
        let event = cx.create_event()?;
        if let Some(slot) = self.groups.get_mut(index) {
            slot.event = Some(event);
        }
        listen(&mut state, event, group, now, true);
        drop(state);
        reply.copy_handle(event);
        reply.ok()
    }

    /// `DuplicateRequestGroupEvent` (32). in: copy handle, the group's event; `u32` group. out:
    /// copy handle, the same event.
    fn duplicate_request_group_event(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let copied = req.copy_handle(0)?;
        let group = req.payload().read_u32()?;
        let index = self.group(group)?;
        let event = self
            .groups
            .get(index)
            .and_then(|g| g.event)
            .ok_or(VI_NOT_FOUND)?;
        if copied != event {
            return Err(KernelError::InvalidHandle.into());
        }
        reply.copy_handle(event);
        reply.ok()
    }

    /// `ConfigureRequestGroups` (51). in: `u32` count.
    fn configure_request_groups(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let count = req.payload().read_u32()?;
        if count >= REQUEST_GROUPS {
            return Err(VI_OUT_OF_RANGE.into());
        }
        if self.group_count.is_some() {
            return Err(VI_PERMISSION_DENIED.into());
        }
        self.group_count = Some(count);
        reply.ok()
    }

    /// `OpenDisplay` (101). in: `u64` name, `u32` flags.
    fn open_display(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let name = payload.read_u64()?;
        let flags = payload.read_u32()?;
        tracing::debug!(name, flags, "gsv:vi2: open display");
        if name != DEFAULT_DISPLAY {
            return Err(KernelError::NotFound.into());
        }
        if flags != OPEN_DISPLAY_FLAGS {
            return Err(KernelError::NotImplemented.into());
        }
        if self.display_references >= MAX_DISPLAY_REFERENCES {
            return Err(KernelError::OutOfResource.into());
        }
        self.display_references += 1;
        reply.ok()
    }

    /// `CloseDisplay` (102). in: `u64` name.
    fn close_display(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let name = req.payload().read_u64()?;
        if player_display(name).is_some() {
            return Err(KernelError::NotFound.into());
        }
        if name != DEFAULT_DISPLAY {
            return Err(KernelError::InvalidCombination.into());
        }
        if !self.is_open() {
            return Err(KernelError::NotFound.into());
        }
        self.display_references -= 1;
        if !self.is_open() {
            self.notify(cx, OPEN_DISPLAY_FLAGS)?;
        }
        reply.ok()
    }

    /// `GetHdrStatus` (104). in: `u64` name. out: `u64` format, `u32` 0. Called about 75 times
    /// a second (row 119): it logs at trace level only.
    fn get_hdr_status(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let name = req.payload().read_u64()?;
        tracing::trace!(name, "gsv:vi2: HDR status");
        if name != DEFAULT_DISPLAY || !self.is_open() {
            return Err(VI_NOT_FOUND.into());
        }
        reply.payload().write_u64(HDR_FORMAT_SDR).write_u32(0);
        reply.ok()
    }

    /// The checks of 105, 107 and 354 (module docs, "Commands"): the name, read from `req`.
    fn known_name(req: &Request<'_>) -> Result<u64, Fault> {
        let name = req.payload().read_u64()?;
        tracing::trace!(name, "gsv:vi2: display state");
        if name == 0 {
            return Err(VI_OUT_OF_RANGE.into());
        }
        if !is_known_display(name) {
            return Err(VI_NOT_FOUND.into());
        }
        Ok(name)
    }

    /// `GetDisplayState105` (105). in: `u64` name. out: `u32` [`DISPLAY_STATE_105`]. Called
    /// about 75 times a second (row 119): it logs at trace level only.
    fn get_display_state_105(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::known_name(req)?;
        reply.payload().write_u32(DISPLAY_STATE_105);
        reply.ok()
    }

    /// `GetBufferMaxResolution` (106). in: `u64` name. out: `u64` mode.
    fn get_buffer_max_resolution(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let name = req.payload().read_u64()?;
        tracing::trace!(name, "gsv:vi2: maximum buffer resolution");
        if name != DEFAULT_DISPLAY || !self.is_open() {
            return Err(VI_NOT_FOUND.into());
        }
        reply.payload().write_u64(BUFFER_RESOLUTION_720P);
        reply.ok()
    }

    /// `GetDisplayState107` (107). in: `u64` name. out: three `u64`, [`DISPLAY_STATE_107`].
    fn get_display_state_107(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::known_name(req)?;
        for value in DISPLAY_STATE_107 {
            reply.payload().write_u64(value);
        }
        reply.ok()
    }

    /// `GetDisplayState354` (354). in: `u64` name. out: `u64` [`DISPLAY_STATE_354`].
    fn get_display_state_354(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::known_name(req)?;
        reply.payload().write_u64(DISPLAY_STATE_354);
        reply.ok()
    }

    /// `OpenEndpoint` (111). in: `u64` name, `u64` display name, `u64` client.
    fn open_endpoint(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let name = payload.read_u64()?;
        let display = payload.read_u64()?;
        let token = payload.read_u64()?;
        let bound = self
            .client(token)
            .and_then(|index| self.clients.get(index))
            .is_some_and(|c| c.bound);
        if !bound || !self.is_open() || name != DEFAULT_DISPLAY || display != name {
            return Err(KernelError::NotFound.into());
        }
        if self.endpoint.is_some() {
            return Err(KernelError::InvalidState.into());
        }
        self.endpoint = Some(token);
        reply.ok()
    }

    /// `CloseEndpoint` (112). in: `u64` name, `u64` display name.
    fn close_endpoint(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let name = payload.read_u64()?;
        let display = payload.read_u64()?;
        if name != DEFAULT_DISPLAY || display != name {
            return Err(KernelError::InvalidCombination.into());
        }
        if self.endpoint.take().is_none() {
            return Err(KernelError::NotFound.into());
        }
        reply.ok()
    }

    /// `RegisterClientState` (201). in: `u64` client, `u32` retire bit, `u32` available bit.
    fn register_client_state(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let retire = payload.read_u32()?;
        let available = payload.read_u32()?;
        tracing::debug!(token, retire, available, "gsv:vi2: register client");
        if token == 0
            || retire >= CLIENT_NOTIFICATIONS
            || available >= CLIENT_NOTIFICATIONS
            || retire == available
        {
            if retire >= CLIENT_NOTIFICATIONS || available >= CLIENT_NOTIFICATIONS {
                tracing::warn!(
                    retire,
                    available,
                    "gsv:vi2: a client notification bit PR refuses; refused"
                );
            }
            return Err(KernelError::InvalidCombination.into());
        }
        let client = Client::new(token, retire, available);
        let held = self.clients.iter().fold(0, |held, c| held | c.bits());
        if self.client(token).is_some() || held & client.bits() != 0 {
            return Err(KernelError::InvalidState.into());
        }
        if self.clients.len() >= MAX_CLIENTS {
            return Err(KernelError::OutOfResource.into());
        }
        let shared = shared(cx)?;
        let registered = shared
            .lock()
            .presents_mut()
            .register(token, available, self.notification_event);
        if let Err(error) = registered {
            tracing::debug!(%error, "gsv:vi2: register client refused");
            return Err(match error {
                RegisterError::Held { .. } => KernelError::InvalidState,
                RegisterError::Full => KernelError::OutOfResource,
            }
            .into());
        }
        self.clients.push(client);
        reply.ok()
    }

    /// `UnregisterClientState` (202). in: `u64` client.
    fn unregister_client_state(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let token = req.payload().read_u64()?;
        if token == 0 {
            return Err(KernelError::InvalidCombination.into());
        }
        let index = self.client(token).ok_or(KernelError::NotFound)?;
        let shared = shared(cx)?;
        let client = self.clients.remove(index);
        if !shared.lock().unregister_present(token) {
            tracing::debug!(token, "gsv:vi2: a retired client had no presents to drop");
        }
        if self.endpoint == Some(token) {
            self.endpoint = None;
        }
        tracing::debug!(token, layer = ?client.layer, "gsv:vi2: client retired");
        self.notify(cx, client.retire)?;
        reply.ok()
    }

    /// `GetClientAvailability` (212). in: `u64` client. out: `u64` its free slots. Makes the
    /// client's layer on first use.
    fn get_client_availability(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let token = req.payload().read_u64()?;
        let shared = shared(cx)?;
        self.refresh(&mut shared.lock());
        if token == 0 {
            return Err(KernelError::InvalidCombination.into());
        }
        let index = self.client(token).ok_or(KernelError::NotFound)?;
        if !self.is_open() {
            return Err(KernelError::NotFound.into());
        }
        let next_layer = self.next_layer;
        let client = self.clients.get_mut(index).ok_or(KernelError::NotFound)?;
        if client.layer.is_none() {
            let after = next_layer
                .checked_add(1)
                .ok_or(KernelError::OutOfResource)?;
            client.layer = Some(next_layer);
            self.next_layer = after;
        }
        let available = client.available;
        reply.payload().write_u64(available);
        reply.ok()
    }

    /// `CancelClientBuffer` (228). in: `u64` client (w2–3), the slot in the low byte of w4 (PR
    /// `DecodeBufferSlot(packet[4])`). out: `u64` free slots, four zero words (module docs,
    /// "Present").
    fn cancel_client_buffer(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let slot = low_byte(payload.read_u32()?);
        if slot >= SLOTS {
            return Err(KernelError::InvalidCombination.into());
        }
        let shared = shared(cx)?;
        self.refresh(&mut shared.lock());
        let client = self.known_client(token)?;
        client.available |= slot_bit(slot);
        let available = client.available;
        tracing::debug!(token, slot, available, "gsv:vi2: cancel buffer");
        write_slots(reply, available);
        reply.ok()
    }

    /// `PresentClientBuffer` (229). in: `u64` client (w2–3), the slot in the low byte of w4, the
    /// native image (w5), the byte offset (w6), the shown extent (w7), the layout (w8), the
    /// fence's `SyncCounterId` (w22), 0 (w23), the fence's value (w24), 0 (w25). out: `u64` free
    /// slots and four zero words, or, under [`Pacing::FenceWait`], a held reply of every slot
    /// (module docs, "Present").
    fn present_client_buffer(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let words = PresentWords::read(req)?;
        let holds = pacing(cx)? == Pacing::BufferQueue;
        let shared = shared(cx)?;
        let mut state = shared.lock();
        self.refresh(&mut state);
        let index = self.client(words.token).ok_or(KernelError::NotFound)?;
        let frame = self.frame(&mut state, index, &words, holds)?;
        let now = if state.vsync().is_stopped() {
            Some(now(cx)?)
        } else {
            None
        };
        let prepared = state.prepare_frame(words.token, frame).map_err(|error| {
            tracing::error!(%error, token = words.token, "gsv:vi2: a present could not be queued");
            match error {
                QueueError::NoClient { .. } => KernelError::InvalidState,
                QueueError::Full { .. } | QueueError::Watch(_) => KernelError::OutOfResource,
            }
        })?;
        tracing::debug!(
            token = words.token,
            slot = frame.slot,
            image = frame.image,
            fence = ?frame.fence,
            "gsv:vi2: present queued"
        );
        if holds {
            state.commit_frame(prepared, now, None);
            drop(state);
            let client = self.clients.get_mut(index).ok_or(KernelError::NotFound)?;
            client.available &= !slot_bit(frame.slot);
            write_slots(reply, client.available);
            return reply.ok();
        }
        match cx.defer() {
            Ok(held) => {
                state.commit_frame(prepared, now, Some(held));
                drop(state);
                reply.deferred()
            }
            Err(fault) => {
                state.abandon_frame(prepared);
                Err(fault)
            }
        }
    }

    /// 229's checks after the client's (module docs, "Present"), in PR `QueueNativeImage`'s
    /// order: the frame client `index` would queue, holding its slot if `holds`.
    fn frame(
        &self,
        state: &mut Locked<'_>,
        index: usize,
        words: &PresentWords,
        holds: bool,
    ) -> Result<NewFrame, Fault> {
        let client = self.clients.get(index).ok_or(KernelError::NotFound)?;
        let slot = words.slot;
        let busy = holds && state.presents_mut().holds(words.token, slot);
        let layer = client
            .layer()
            .filter(|_| client.bound && slot < SLOTS && !busy)
            .ok_or(KernelError::InvalidState)?;
        let image = state
            .native_objects()
            .image(words.image)
            .filter(|image| image.window_pool())
            .copied()
            .ok_or(KernelError::NotFound)?;
        let preference = image.preference();
        let (shown_width, shown_height) = extent(words.shown);
        let (width, height) = extent(preference[WORD_EXTENT]);
        let supported = words.offset == 0
            && shown_width <= width
            && shown_height <= height
            && words.layout == PRESENT_LAYOUT
            && preference[WORD_BLOCK_HEIGHT] == PRESENT_BLOCK_HEIGHT
            && PRESENT_FORMATS.contains(&preference[WORD_FORMAT]);
        if !supported {
            return Err(KernelError::NotImplemented.into());
        }
        let fence = u8::try_from(!words.fence_id)
            .ok()
            .and_then(SyncpointId::new)
            .filter(|&id| state.syncpoint_allocated(id))
            .map(|id| Fence {
                id,
                value: words.fence_value,
            });
        let fenced = words.fence_id != NO_FENCE;
        if words.fence_high != 0 || words.fence_tail != 0 || fenced && fence.is_none() {
            return Err(KernelError::InvalidCombination.into());
        }
        let address = state
            .native_objects()
            .backing(image.memory())
            .and_then(|memory| memory.binding())
            .map(|binding| binding.address)
            .ok_or(KernelError::InvalidState)?;
        let dimension = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
        Ok(NewFrame {
            slot,
            image: words.image,
            request: PresentRequest {
                image: ImageDesc {
                    address,
                    width: dimension(width),
                    height: dimension(height),
                    pitch: preference[WORD_PITCH],
                    block_height_log2: preference[WORD_BLOCK_HEIGHT],
                    format: preference[WORD_FORMAT],
                },
                shown: Rect {
                    x: 0,
                    y: 0,
                    width: dimension(shown_width),
                    height: dimension(shown_height),
                },
                layer,
                display: DisplayId(DEFAULT_DISPLAY),
            },
            fence,
            holds_slot: holds,
        })
    }

    /// `BindClientDisplay` (251). in: `u64` client, `u64` name.
    fn bind_client_display(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let name = payload.read_u64()?;
        let open = self.is_open();
        let client = self.known_client(token)?;
        if player_display(name).is_some() {
            return Err(KernelError::InvalidState.into());
        }
        if name != DEFAULT_DISPLAY || !open {
            return Err(KernelError::NotFound.into());
        }
        if client.bound {
            return Err(KernelError::InvalidState.into());
        }
        client.bound = true;
        reply.ok()
    }

    /// `UnbindClientDisplay` (252). in: `u64` client.
    fn unbind_client_display(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let token = req.payload().read_u64()?;
        let client = self.known_client(token)?;
        if !client.bound {
            return Err(KernelError::NotFound.into());
        }
        client.bound = false;
        reply.ok()
    }

    /// 261 to 272: register (odd) or unregister (even) request group `group` as a display
    /// monitor of the kind the command names. in: `u64` name, `u32` group (module docs,
    /// "Commands" and "Vsync").
    fn monitor(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let command = req.command_id();
        let Some((kind, registers)) = monitor_kind(command) else {
            // The table lists 261 to 272 only.
            tracing::error!(command, "gsv:vi2: a monitor command outside 261 to 272");
            return Err(KernelError::InvalidState.into());
        };
        let mut payload = req.payload();
        let name = payload.read_u64()?;
        let group = payload.read_u32()?;
        tracing::debug!(command, name, group, kind, "gsv:vi2: display monitor");
        if registers {
            self.register_monitor(cx, name, group, kind)?;
        } else {
            self.unregister_monitor(cx, name, group, kind)?;
        }
        reply.ok()
    }

    /// Registers `group` as a monitor of display `name`'s events of `kind` (PR
    /// `RegisterRequestGroupMonitor`).
    fn register_monitor(
        &mut self,
        cx: &mut Context<'_>,
        name: u64,
        group: u32,
        kind: u32,
    ) -> Result<(), Fault> {
        let index = self.group(group)?;
        if name == 0 {
            return Err(VI_OUT_OF_RANGE.into());
        }
        let slot = self.groups.get(index).copied().unwrap_or_default();
        let event = slot
            .event
            .filter(|_| is_known_display(name))
            .ok_or(VI_NOT_FOUND)?;
        if slot.monitor.is_some() {
            return Err(VI_PERMISSION_DENIED.into());
        }
        let now = if kind == VSYNC_KIND {
            Some(now(cx)?)
        } else {
            None
        };
        let shared = shared(cx)?;
        let mut state = shared.lock();
        if let Some(slot) = self.groups.get_mut(index) {
            slot.monitor = Some(Monitor {
                display: name,
                kind,
            });
        }
        if let Some(now) = now {
            listen(&mut state, event, group, now, false);
        } else {
            state.unlisten_vsync(event);
            cx.signal_event(event)?;
        }
        drop(state);
        Ok(())
    }

    /// Unregisters `group` if it is registered as a monitor of display `name`'s events of
    /// `kind` (PR `UnregisterRequestGroupMonitor`); anything else changes nothing.
    fn unregister_monitor(
        &mut self,
        cx: &mut Context<'_>,
        name: u64,
        group: u32,
        kind: u32,
    ) -> Result<(), Fault> {
        let wanted = Monitor {
            display: name,
            kind,
        };
        let Some(index) = self.group(group).ok() else {
            return Ok(());
        };
        let Some(slot) = self.groups.get(index).copied() else {
            return Ok(());
        };
        let (Some(event), true) = (slot.event, name != 0 && slot.monitor == Some(wanted)) else {
            return Ok(());
        };
        let now = now(cx)?;
        let shared = shared(cx)?;
        let mut state = shared.lock();
        if let Some(slot) = self.groups.get_mut(index) {
            slot.monitor = None;
        }
        listen(&mut state, event, group, now, true);
        cx.signal_event(event)?;
        drop(state);
        Ok(())
    }

    /// `SetDisplayParameter301` (301). in: `u64` name, `u64` value, `u64` mask.
    fn set_display_parameter_301(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let name = payload.read_u64()?;
        let value = payload.read_u64()?;
        let mask = payload.read_u64()?;
        tracing::debug!(name, value, mask, "gsv:vi2: display parameter 301");
        if name != DEFAULT_DISPLAY {
            return Err(KernelError::InvalidCombination.into());
        }
        reply.ok()
    }

    /// `SetClientOutputEncoding` (351). in: `u64` client, `u64` encoding.
    fn set_client_output_encoding(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let encoding = payload.read_u64()?;
        self.known_client(token)?;
        if encoding != OUTPUT_ENCODING {
            tracing::warn!(encoding, "gsv:vi2: an output encoding PR refuses; refused");
            return Err(KernelError::NotImplemented.into());
        }
        reply.ok()
    }

    /// `SetClientOutputLevels` (352). in: `u64` client, three `f32`. Levels it refuses answer
    /// [`VI_OUT_OF_RANGE`] (module docs, "Commands": a **conflict**).
    fn set_client_output_levels(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let token = payload.read_u64()?;
        let bits = [
            payload.read_u32()?,
            payload.read_u32()?,
            payload.read_u32()?,
        ];
        let client = self.known_client(token)?;
        if !levels_are_valid(bits.map(f32::from_bits)) {
            return Err(VI_OUT_OF_RANGE.into());
        }
        client.levels = Some(bits);
        reply.ok()
    }
}

/// The interfaces a vi2 server's sessions can be.
#[must_use]
pub fn vi2_interfaces() -> Interfaces {
    Interfaces::new().with::<Vi2Session>()
}

/// The vi2 server over gsv's `shared` state, pacing with the buffer queue: every session a
/// [`Vi2Session`], sharing a [`Vi2Family`]. Give it the run's environment (`Server::with_env`):
/// a group starting to listen to the vsync clock, and a present on a stopped clock, read its
/// clock (module docs, "The server").
#[must_use]
pub fn vi2_server(policy: UnknownPolicy, shared: Arc<Shared>) -> Server {
    vi2_server_paced(policy, shared, Pacing::BufferQueue)
}

/// The vi2 server of [`vi2_server`], pacing with `pacing` (the per-run setting of the owner's
/// decision 1).
#[must_use]
pub fn vi2_server_paced(policy: UnknownPolicy, shared: Arc<Shared>, pacing: Pacing) -> Server {
    Server::new(VI2_NAME, policy)
        .with_root(|| Box::new(Vi2Session::new()))
        .with_interfaces(vi2_interfaces())
        .with_family(Vi2Family::new(shared).with_pacing(pacing))
}
