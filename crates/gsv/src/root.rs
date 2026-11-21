//! The graphics server's root: the session `am`'s `ISelfController` 2000 hands a title, its
//! bootstrap, and every other root command as a logged stub with its known reply shape
//! (`docs/specs/graphics-server.md`, "The root"; port slice 1).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` (`IGraphicsSupervisor`)
//! and `graphics_supervisor_protocol.h` @ e5fb3b97, and Eden-NX2,
//! `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb (`INx2GraphicsSystemSession`), both
//! GPL-3.0-or-later; "PR" and "NX2" below, as in [`protocol`](crate::protocol), which holds the
//! command table, its marks and sources.
//!
//! # The server
//!
//! [`root_server`] is the root's server, one of the graphics server's per-family servers
//! (decision 6). It is a private entry of the service registry under [`ROOT_NAME`], which no
//! title can open through `sm:`; whoever puts the services together gives its number to `am`
//! (`starquake_services::am`, "The graphics server"). Every session on it is a [`GsvRoot`], and
//! all of them share one [`RootState`], the server's family state. This matches PR, where `am`'s
//! `ISelfController` keeps the root session it made on the first 2000
//! (`m_graphics_supervisor_session`) and hands that same session out on every later one, so
//! there is one root state. It does not match NX2: `am` makes a new
//! `INx2GraphicsSystemSession` on every 2000
//! (`eden-nx2/src/core/hle/service/am/service/self_controller.cpp` @ 50fa86bb,
//! `GetNx2GraphicsSystemSession`), and each session owns its own `m_objects` and
//! `m_syncpt_events` (`eden-nx2/src/core/hle/service/nx2_gsv.h`), so a second NX2 session
//! starts fresh. Here a second root session sees the first one's phase, out-of-order count and
//! 204 events (listed under "Where this differs from the references").
//!
//! Every command is TIPC, as in both references (PR `RegisterHandlersTipc`; NX2
//! `RegisterHandlersTipc`): a CMIF request (`a_cmif_request_to_the_root_goes_to_the_policy`),
//! and a TIPC command the table does not list, is answered by the server's unknown-command
//! policy.
//!
//! # The bootstrap
//!
//! | Id | Command | In | Out |
//! |---|---|---|---|
//! | 4 | `Initialize` | copy handle 0: the process; `u64` applet resource user id, `u64` flags, four `u64` parameters | nothing |
//! | 11 | `BindProcess` | copy handle 0: the process | nothing |
//! | 101 | `CompleteBootstrap` | nothing | nothing |
//! | 126 | `QueryMemoryMode` | nothing | `u32` 0: no remote memory |
//! | 115 | `ConfigureNativeMemory` | `u64` mode (guess: PR checks two words, the first 0 or 2 and the second 0, read here as one `u64`) | `u64` 0 |
//!
//! The layouts are PR's: `DecodeBootstrap` (words 3, 4–5, 6–7 and 8–15 after a header of
//! `0x14, 0x8000000c, 2`: one copied handle and twelve data words), `BindProcess` (`0x1b,
//! 0x80000000, 2`: one copied handle, no data), `CompleteBootstrap` (`0x75, 0`), and
//! `QueryMemoryMode` and `ConfigureNativeMemory` (`rb.Push<u32>(0)`, `rb.Push<u64>(0)`). A
//! bootstrap is local when its flags are one of [`bootstrap::LOCAL_FLAGS`] and every parameter
//! is zero (PR `IsLocalBootstrap`); its GPU profile is [`bootstrap::gpu_profile`] (PR
//! `NativeGpuProfileForBootstrap`). Both are recorded with the request ([`BootstrapRequest`]),
//! and handed to the GPU by a later slice.
//!
//! **Order (decision 7: log first, enforce once a trace confirms it).** PR keeps a phase machine
//! (`Created` → 4 → `Initialized` → 11 → `ProcessBound` → 101 → `Ready`) and answers a
//! bootstrap command in any other phase with `Kernel::ResultInvalidState`; NX2 accepts every
//! command. Here the phase is kept as PR keeps it ([`Phase`]), and a bootstrap command found in
//! a phase other than the one PR requires is logged, counted ([`RootState::out_of_order`]) and
//! answered with success. The phase it moves to is reached either way, so later commands are
//! judged against the order that actually happened.
//!
//! # Stubs
//!
//! Every other command of the table ([`crate::protocol::ROOT_COMMANDS`]), and in a root without
//! gsv's shared state the native ones (below, "Native memory and images"), is a logged stub that
//! replies success with its known reply shape ([`crate::protocol::ReplyShape`]): its handles,
//! then its data words, all zero. Each handle is copied ([`crate::protocol::HANDLE_KIND`]):
//!
//! - **A session** is a new session on this server, a [`Placeholder`] that has no commands, so
//!   every request to it is answered by the server's unknown-command policy, logged. NX2 began
//!   the same way (`INx2NvsrvSession`, "a placeholder ... every command gets an empty success
//!   and is logged"). The family's slice replaces it with a session on its own server: 202,
//!   497 and 1402 do in a root wired to their servers (below, "The family sessions"), 305 in
//!   one wired to the vi2 server (below, "The display session (305)"), and 902, 906 and 1002
//!   in one wired to the video server (below, "Video (slice 15)").
//! - **An event** (only 204's two) is a kernel event, unsignalled. They are made the first time
//!   204 is asked and kept ([`RootState::syncpoint_events`]), so a repeated 204 hands out the
//!   same two and a failure between the two keeps the first. A root built with gsv's shared
//!   state answers 204 with the shared page instead (below).
//!
//! # The shared syncpoint page (204)
//!
//! A root built for a run with gsv's shared state ([`root_server_with_page`]) answers 204 by
//! binding the shared syncpoint page ([`crate::shared::page`], "Binding (204)"), whose state is
//! in gsv's shared state, not the root's: the drain and the page turn act on it. The root's
//! family ([`RootFamily`]) holds the handle to that state beside [`RootState`]; the handle is the
//! run's wiring, so a snapshot holds only the [`RootState`] and loading keeps the handle. Before
//! the bootstrap is complete 204 is logged and accepted, as the stubs are (decision 7). 204
//! keeps a reference of the root's own to each of the page's two transfer memories; when the
//! server's last session closes, the family queues
//! [`Completion::RootClosed`](crate::completion::Completion) and the timer thread's drain lets
//! the page and both references go if no root session is open by then
//! ([`crate::shared::page`], "When the root closes").
//!
//! # The gpu session (497)
//!
//! A root built for a run with a gpu server ([`RootState::with_gpu_server`], [`root_server_for`])
//! answers 497 with a copied session on that server ([`Reply::copy_service`]), whose root
//! interface is the gpu session (`crate::gpu`). 497's `u64` is read: PR requires 4
//! (`OpenGpuSession`: `flags == 4`, else `ResultInvalidCombination`) and the bootstrap complete
//! (`ResultInvalidState`); here another value, or a bootstrap not complete, is logged and
//! accepted, as for the other root commands (decision 7). A root with no gpu server answers 497
//! with the stub's placeholder session. The server's number is the run's wiring, as `am`'s
//! graphics server number is: a snapshot that names another one is refused
//! (`a_snapshot_with_another_gpu_server_is_refused`).
//!
//! # Native memory and images
//!
//! A root built for a run with gsv's shared state ([`root_server_with_page`]) answers 111, 112,
//! 114, 314, 315, 423, 429, 430, 435 and 1510 from the native tables in that state
//! ([`crate::native`], which lists each command's words, checks and divergences); one built
//! without answers them with the stubs. 115 stays the root's own and keeps recording its mode.
//! Before the bootstrap is complete each is logged and accepted, as the stubs are (decision 7).
//!
//! # The family sessions (202, 497, 1402)
//!
//! 202 and 1402 hand out sessions on the syncpt server ([`RootState::with_syncpt_server`],
//! [`crate::syncpt`]) and the nvsrv server ([`RootState::with_nvsrv_server`], [`crate::nvsrv`])
//! as 497 does on the gpu server: one session handle and no data (observed: rows 19 and 23; PR
//! `GetSyncpointSession`, `GetNvidiaDriver`; NX2 `OpenSyncptSession`, `OpenNvsrvSession`, which
//! put the handle at word 3 and the result at word 4).
//!
//! **conflict: the handle's kind.** PR moves the session (`rb.PushMoveObjects(ctx, client)` in
//! both); NX2 copies it (`ReturnRawSession`: `rb.PushCopyObjects(ctx, *client)`). Neither
//! answer contains the other, so by decision 8 the one a title ran on wins: NX2's, a copied
//! session ([`Reply::copy_service`]), as for 497 and every root handle
//! ([`crate::protocol::HANDLE_KIND`]). A trace of the reply's handle descriptor settles it.
//!
//! Neither reads a data word. PR checks the header and its state differently for the two: 202
//! answers `ResultInvalidState` unless the header is `0xda, 0`, 201 has been sent
//! (`m_syncpoint_context_initialized`) and so has 1402 (`m_nv_module`), all in one condition,
//! so a bad header on 202 is `ResultInvalidState` too; 1402 answers `ResultInvalidCombination`
//! for a header other than `0x58a, 0`, then `ResultInvalidState` before the bootstrap is
//! complete (`m_phase != Phase::Ready`) or the image formats are set up
//! (`m_image_formats_initialized`). NX2 checks none of it. Here none of it is checked, as the
//! root keeps none of that state: any header that reaches the command is accepted, and a
//! command before the bootstrap is complete is logged and accepted (decision 7). A root with no
//! server for the family answers with the stub's placeholder. Each server's number is
//! the run's wiring, checked on a snapshot as the gpu server's is, the gpu server's first, then
//! the syncpt server's, then the nvsrv server's
//! (`a_snapshot_with_other_family_servers_is_refused_gpu_first`).
//!
//! # The display session (305)
//!
//! 305 hands out a session on the vi2 server ([`RootState::with_vi2_server`], [`crate::vi2`]) as
//! 497 does on the gpu server: one copied session handle and no data (observed: row 21; PR
//! `OpenDisplaySession`; NX2 `OpenViSession`, which puts the handle at word 3 and the result at
//! word 4). Its `u64` is read: PR requires one of [`DISPLAY_SESSION_FLAGS`]
//! (`IsSupportedDisplaySession`, else `ResultInvalidCombination`) and 301 sent before it
//! (`m_display_context_initialized`, else `ResultInvalidState`); here other flags are logged and
//! accepted (decision 7: log first), the root keeps no state of 301 so it is not checked, and a
//! 305 before the bootstrap is complete is logged and accepted, as for the other family
//! commands. A root with no vi2 server answers 305 with the stub's placeholder. The server's
//! number is the run's wiring, checked on a snapshot after the nvsrv server's
//! (`a_snapshot_with_another_vi2_server_is_refused`).
//!
//! # Video (slice 15)
//!
//! A root built for a run with a video server ([`RootState::with_video_server`]) answers 902,
//! 906 and 1002 with a copied session on it, whose root interface is the video context
//! ([`crate::video`]), as 497 does on the gpu server; one with no video server answers them
//! with the stub's placeholder. 902 and 1002 read nothing; 906 reads a `u8` (word 2) and a
//! `u64` library version (words 3–4) (PR `DecodeVideoContext`: `packet[2] > 1` is
//! `InvalidCombination`, then a remote-memory flag or a version other than
//! [`VIDEO_LIBRARY_VERSION`] is `NotImplemented`; NX2 reads nothing, row 146). Here they are
//! logged and accepted, the larger answer by decision 8, and so is any of the three before 901
//! (PR: `InvalidState`; decision 7). 1002 is the **conflict** the protocol table names: its
//! context is the same as 902's ([`crate::video`], "The server"). The 906 values PR refuses are
//! only logged, a warning, which no test captures: its mutation (dropping the warning) is not
//! observable but in the log. Pinned in `root/tests/video.rs`.
//!
//! 901, 903, 1001 and 1003 stay the stubs: success, nothing kept. PR keeps whether each was
//! sent, to refuse the other video commands without it (`InvalidState`), which decision 7 logs
//! rather than refuses here; NX2 stubs 901 and 1001 and has no 903 or 1003.
//!
//! A root built with gsv's shared state ([`root_server_with_page`]) answers 991, 911, 912, 961,
//! 426, 427 and 428 from the native tables ([`crate::native::pools`], which lists each
//! command's words, checks and divergences), and 602 with a camera session
//! ([`crate::video::camera`]); one built without answers them with the stubs. Before the
//! bootstrap is complete each is logged and accepted, as the native commands are.
//!
//! # Restoring
//!
//! A snapshot is a file, so a restore believes nothing in the [`RootState`]. In this order, the
//! first that fails refusing the snapshot and leaving the server as it was:
//!
//! 1. The relations between its fields, as it decodes ([`RootStateError`]; refused with
//!    `SnapshotError::Decode(postcard::Error::SerdeDeCustom)`), 204's two events being two
//!    objects among them (`a_saved_state_no_command_reaches_is_refused`).
//! 2. Beside the kernel the server is restored beside (`Server::restore_beside`), 204's
//!    events: each kept is a live readable event the root's server made, under the number it is
//!    known by, as 204 makes them (`Context::create_event` returns an event's readable end), the
//!    first before the second; the first that fails is refused with
//!    [`SnapshotError::FamilyObject`] naming it (`each_syncpoint_event_restores_only_as_one_the_root_made`,
//!    `the_first_syncpoint_event_is_checked_first`, `a_root_s_own_events_restore_and_carry_on`,
//!    in `root/tests/restore.rs`). The server keeps kernel objects, so `Server::restore`, which
//!    has no kernel, refuses its snapshots with [`SnapshotError::NeedsKernel`].
//! 3. The gpu, syncpt, nvsrv, vi2 and video server numbers, in that order, as the run's wiring
//!    ([`SnapshotError::FamilyWiring`]; "The family sessions"). They come after the events
//!    (`the_events_are_checked_before_the_server_numbers`): the events are checked by the
//!    server's check, which runs before the family loads, and the numbers as it loads
//!    ([`RootFamily`]).
//!
//! Not checked here:
//!
//! - **The bootstrap's applet resource user id, and the other recorded words.** The root does
//!   not check them live either ("Where this differs from the references"); none names a kernel
//!   object.
//! - **Which of the root's own events is which.** The kernel notes that the root's server made
//!   an event, not which command made it, so a snapshot may swap 204's two events, or name in
//!   them events the root made for another purpose (the shared page's answer event, in a root
//!   built with one, is the page's: [`crate::shared::page`] checks it). 204 then hands out the
//!   swapped or borrowed events.
//! - **Whether the events' signalled state agrees with anything.** It is the kernel's, saved with
//!   the kernel; 204's events are never signalled here.
//! - **Which server made an event, across snapshots.** The kernel's IPC table, which holds that
//!   note, has no snapshot of its own yet, so today a root restores only beside the tables it
//!   ran on (`starquake_services::framework`'s `restored` module).
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `root/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | A bootstrap command out of PR's order | `ResultInvalidState` | success | success, logged and counted (`bootstrap_out_of_order_is_logged_not_refused`) |
//! | A bootstrap that asks for remote memory (flags not local, or a parameter not zero) | `ResultNotImplemented` | success | **conflict**: success, the request recorded with its mode, logged; verification 5 (`a_remote_memory_bootstrap_is_accepted_and_recorded`) |
//! | 4's or 11's copied handle names an object that is not the process the root was made for, or 4's applet resource user id is not the caller's | `ResultInvalidHandle`, `ResultInvalidArgument` | success | success: the root knows neither the process nor the applet resource user id; only a handle that names nothing is refused, with the kernel's error (`the_bootstrap_takes_any_object_and_any_user_id`) |
//! | A request whose header is not PR's exact one | `ResultInvalidCombination` | read as it comes | a request that lacks the handle or data a command reads is `INVALID_CMIF_REQUEST`, the handle checked first; one with more data words is read (`a_short_bootstrap_request_is_refused_and_a_longer_one_is_read`, `the_handle_is_checked_before_the_data`) |
//! | 115 with a mode other than 0 or 2 | `ResultInvalidCombination` | not in its table: auto-stubbed, success | success, the mode recorded, logged (`configure_native_memory_records_any_mode`) |
//! | 126, 115 and the stubs before the bootstrap is complete | `ResultInvalidState` for most | success | success, the phase logged (`every_root_command_s_reply_bytes`, sent before any bootstrap) |
//! | A TIPC command the table does not list | `ResultNotImplemented` | auto-stubbed | the server's unknown-command policy (`a_command_the_table_does_not_list_goes_to_the_policy`) |
//! | A second root session (a second 2000) | the same session as the first: one state | a new session with its own `m_objects` and `m_syncpt_events` | a new session that shares the first one's [`RootState`]: its phase, its out-of-order count and 204's two events (`two_root_sessions_share_the_phase_and_the_syncpoint_events`); with the shared page, the one page (`a_second_bind_is_refused_and_changes_nothing`, in `shared/page/tests.rs`) |
//! | The handle 202 and 1402 hand out | moved | copied | **conflict**: copied, NX2's (`a_family_session_reply_is_one_copied_handle_and_success`) |
//! | 202 before 201 and 1402, or 1402 before the bootstrap is complete and the image formats are set up | `ResultInvalidState` | success | success (`each_family_command_hands_out_a_session_on_its_own_server`) |
//! | 202 or 1402 whose header is not PR's exact one (here: a data word) | `ResultInvalidState` (202), `ResultInvalidCombination` (1402) | success | success: the data is not read (`a_family_command_with_a_data_word_is_accepted`) |
//! | 305 with flags other than [`DISPLAY_SESSION_FLAGS`], or before 301 | `ResultInvalidCombination`, `ResultInvalidState` | success, the flags logged | success, the flags logged (decision 7; `open_display_session_copies_a_session_on_the_vi2_server`) |
//! | 902, 906 or 1002 before 901 (or 1001), or before the bootstrap is complete | `ResultInvalidState` | success | success (`each_video_open_hands_out_a_session_on_the_video_server`) |
//! | 906 with a remote-memory byte not 0, or a library version not [`VIDEO_LIBRARY_VERSION`] | `ResultInvalidCombination` (a byte above 1), `ResultNotImplemented` | success | success, logged (`each_video_open_hands_out_a_session_on_the_video_server`) |
//! | The handle 902, 906 and 1002 hand out | moved | copied | **conflict**: copied, NX2's (`a_video_open_reply_is_one_copied_handle_and_success`) |
//! | 1002 | the encoder's context, whose codec refuses to initialise as a decoder | a second decoder | **conflict**: the same context as 902's (`the_root_hands_out_video_sessions_that_count_frames`, in `starquake-core`'s `hle/tests/graphics/video.rs`) |
//! | 901, 903, 1001 and 1003 | kept: whether each video service is initialised | 901 and 1001 stubbed; no 903 or 1003 | the stubs: success, nothing kept (`every_root_command_s_reply_bytes`) |
//! | 602 with a device type above 1 | `0x280f3` | (no 602) | `0x280f3` (`open_camera_session_takes_device_types_0_and_1`, in `video/camera/tests.rs`) |
//! | 601 and 602 before the bootstrap is complete | `ResultInvalidState` | (no 601 or 602) | success, logged (decision 7) |

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, NOT_SUPPORTED, Outcome, Reply, Request, RestoredKernel,
    Server, SnapshotError, UnknownPolicy,
};
use starquake_trace::ServiceId;

use crate::completion::Completion;
use crate::gpu::low_byte;
use crate::native;
use crate::native::pools;
use crate::protocol::command::{
    ACTIVATE_FRAME_CONTEXT, ACTIVATE_GPU_CONTEXT, BIND_NATIVE_GPU_MEMORY, BIND_NATIVE_MEMORY,
    BIND_PROCESS, BIND_SYNCPOINT_MEMORY, CALCULATE_IMAGE_REQUIREMENTS, CALCULATE_NATIVE_IMAGE_SIZE,
    CALCULATE_REQUIRED_BUFFER_MEMORY, CALCULATE_VIDEO_IMAGE_REQUIREMENTS, COMPLETE_BOOTSTRAP,
    CONFIGURE_NATIVE_MEMORY, CREATE_CAMERA_CONTEXT, CREATE_CHAT_IMAGE_POOL, CREATE_NATIVE_IMAGE,
    CREATE_NATIVE_MEMORY, CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, CREATE_VIDEO_IMAGE_POOL,
    DESTROY_NATIVE_IMAGE, DESTROY_NATIVE_MEMORY, DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL,
    DESTROY_VIDEO_IMAGE_POOL, FINALIZE_DISPLAY_CONTEXT, FINALIZE_VIDEO_DECODER,
    FINALIZE_VIDEO_ENCODER, GET_NATIVE_EVENT_1121, GET_NVIDIA_DRIVER, GET_SYNCPOINT_SESSION,
    INITIALIZE, INITIALIZE_DISPLAY_CONTEXT, INITIALIZE_FRAME_CONTEXT, INITIALIZE_GPU_CONTEXT,
    INITIALIZE_IMAGE_CONTEXT_1501, INITIALIZE_IMAGE_CONTEXT_1502, INITIALIZE_SYNCPOINT_CONTEXT,
    INITIALIZE_VIDEO_DECODER, INITIALIZE_VIDEO_ENCODER, MAP_NATIVE_GPU_IMAGE, OPEN_CAMERA_SESSION,
    OPEN_DISPLAY_SESSION, OPEN_GPU_SESSION, OPEN_VIDEO_DECODER_CONTEXT,
    OPEN_VIDEO_DECODER_CONTEXT_902, OPEN_VIDEO_ENCODER_CONTEXT, QUERY_MEMORY_MODE,
    QUERY_NATIVE_GPU_IMAGE, QUERY_VIDEO_IMAGE, REGISTER_GPU_ADDRESS_SPACE, SEQUENCE_COMMAND_1201,
    SEQUENCE_COMMAND_1202, SEQUENCE_COMMAND_1203, SET_GRAPHICS_CLASS_1111, UNMAP_NATIVE_GPU_IMAGE,
};
use crate::protocol::{Handed, bootstrap, root_command};
use crate::shared::Shared;
use crate::shared::page;
use crate::video::camera::{self, CameraSession};

#[cfg(test)]
mod tests;

/// The name of the root's private entry in the service registry. Starquake's own: on a console
/// `am` makes the session itself (PR `ISelfController::GetGraphicsSupervisor`), so the root has
/// no registered name in any reference (`starquake_services::sm`, "Private entries").
pub const ROOT_NAME: &str = "gsv";

/// Where the bootstrap is, as PR's phase machine names it (`IGraphicsSupervisor::Phase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Before `Initialize` (4).
    Created,
    /// After `Initialize`.
    Initialized,
    /// After `BindProcess` (11).
    ProcessBound,
    /// After `CompleteBootstrap` (101): every other command may run.
    Ready,
}

impl Phase {
    /// The phase as a word, for logs.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Initialized => "initialized",
            Self::ProcessBound => "process bound",
            Self::Ready => "ready",
        }
    }

    /// For a bootstrap command, the phase PR requires it in and the phase it moves to
    /// (PR `Initialize`: `m_phase != Phase::Created`; `BindProcess`: `m_phase !=
    /// Phase::Initialized`; `CompleteBootstrap`: `m_phase != Phase::ProcessBound`).
    #[must_use]
    pub const fn step(command: u32) -> Option<(Self, Self)> {
        match command {
            INITIALIZE => Some((Self::Created, Self::Initialized)),
            BIND_PROCESS => Some((Self::Initialized, Self::ProcessBound)),
            COMPLETE_BOOTSTRAP => Some((Self::ProcessBound, Self::Ready)),
            _ => None,
        }
    }
}

/// What `Initialize` (4) was asked (PR `BootstrapRequest`, without the process handle, which
/// names an object only for the request's length).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapRequest {
    /// The applet resource user id.
    pub applet_resource_user_id: u64,
    /// The flags.
    pub flags: u64,
    /// The parameters after the flags.
    pub parameters: [u64; bootstrap::PARAMETERS],
}

impl BootstrapRequest {
    /// Whether it is a local bootstrap (no remote memory): PR `IsLocalBootstrap`.
    #[must_use]
    pub fn is_local(&self) -> bool {
        bootstrap::is_local(self.flags, &self.parameters)
    }

    /// The GPU profile its flags ask for: PR `NativeGpuProfileForBootstrap`.
    #[must_use]
    pub const fn gpu_profile(&self) -> Option<u32> {
        bootstrap::gpu_profile(self.flags)
    }
}

/// The events the 204 stub has made, in the order it hands them out. Kept so that a repeated
/// 204 hands out the same two, and so that the first is not forgotten when making the second
/// fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncpointEvents {
    /// None made yet.
    #[default]
    None,
    /// The first made; making the second failed.
    One(ObjectId),
    /// Both made.
    Both(ObjectId, ObjectId),
}

/// What every root session shares: the root server's family state. Plain data, saved in the
/// server's snapshot. Deserialising goes through `SavedRootState`, which refuses a state the
/// root's commands cannot reach ([`RootStateError`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedRootState")]
pub struct RootState {
    phase: Phase,
    bootstrap: Option<BootstrapRequest>,
    out_of_order: u32,
    native_memory_mode: Option<u64>,
    syncpoint_events: SyncpointEvents,
    gpu_server: Option<ServiceId>,
    syncpt_server: Option<ServiceId>,
    nvsrv_server: Option<ServiceId>,
    vi2_server: Option<ServiceId>,
    video_server: Option<ServiceId>,
}

/// The serialised form of [`RootState`], field for field, and what deserialising validates.
#[derive(Deserialize)]
struct SavedRootState {
    phase: Phase,
    bootstrap: Option<BootstrapRequest>,
    out_of_order: u32,
    native_memory_mode: Option<u64>,
    syncpoint_events: SyncpointEvents,
    gpu_server: Option<ServiceId>,
    syncpt_server: Option<ServiceId>,
    nvsrv_server: Option<ServiceId>,
    vi2_server: Option<ServiceId>,
    video_server: Option<ServiceId>,
}

/// Why a saved [`RootState`] could not be read back: it holds a combination of fields that no
/// sequence of root commands leads to. The phase only ever leaves [`Phase::Created`] (no
/// command steps back to it), only `Initialize` (4) enters [`Phase::Initialized`] and records
/// a bootstrap, the first of 11 and 101 sent before any 4 is out of PR's order, and 204 makes
/// two distinct events. These are every relation between the fields; the native memory mode
/// and a lone 204 event may be anything in any phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RootStateError {
    /// [`Phase::Created`] with a bootstrap recorded: 4 moves the phase out of `Created`.
    #[error("a root that no bootstrap command has reached has a recorded bootstrap")]
    CreatedWithBootstrap,
    /// [`Phase::Created`] with out-of-order commands counted: each counted command moves the
    /// phase out of `Created`.
    #[error("a root that no bootstrap command has reached has counted {count} out of order")]
    CreatedWithCount {
        /// The count in the data.
        count: u32,
    },
    /// [`Phase::Initialized`] with no bootstrap recorded: only 4 leads there, and it records
    /// one.
    #[error("an initialized root has no recorded bootstrap")]
    InitializedWithoutBootstrap,
    /// Past [`Phase::Created`] with no bootstrap and nothing counted: with no 4 sent, the
    /// phase left `Created` through 11 or 101, which are out of order there.
    #[error(
        "a root that left the created phase without a bootstrap has counted nothing out of order"
    )]
    UncountedWithoutBootstrap,
    /// 204's two events are one object: it makes two.
    #[error("the two syncpoint events are the same object")]
    SameSyncpointEvents,
}

impl TryFrom<SavedRootState> for RootState {
    type Error = RootStateError;

    fn try_from(saved: SavedRootState) -> Result<Self, Self::Error> {
        let SavedRootState {
            phase,
            bootstrap,
            out_of_order,
            native_memory_mode,
            syncpoint_events,
            gpu_server,
            syncpt_server,
            nvsrv_server,
            vi2_server,
            video_server,
        } = saved;
        match (phase, bootstrap.is_some()) {
            (Phase::Created, true) => return Err(RootStateError::CreatedWithBootstrap),
            (Phase::Created, false) if out_of_order != 0 => {
                return Err(RootStateError::CreatedWithCount {
                    count: out_of_order,
                });
            }
            (Phase::Initialized, false) => {
                return Err(RootStateError::InitializedWithoutBootstrap);
            }
            (Phase::ProcessBound | Phase::Ready, false) if out_of_order == 0 => {
                return Err(RootStateError::UncountedWithoutBootstrap);
            }
            _ => {}
        }
        if let SyncpointEvents::Both(first, second) = syncpoint_events
            && first == second
        {
            return Err(RootStateError::SameSyncpointEvents);
        }
        Ok(Self {
            phase,
            bootstrap,
            out_of_order,
            native_memory_mode,
            syncpoint_events,
            gpu_server,
            syncpt_server,
            nvsrv_server,
            vi2_server,
            video_server,
        })
    }
}

impl Default for RootState {
    fn default() -> Self {
        Self::new()
    }
}

impl RootState {
    /// A root no command has reached: [`Phase::Created`], nothing recorded.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: Phase::Created,
            bootstrap: None,
            out_of_order: 0,
            native_memory_mode: None,
            syncpoint_events: SyncpointEvents::None,
            gpu_server: None,
            syncpt_server: None,
            nvsrv_server: None,
            vi2_server: None,
            video_server: None,
        }
    }

    /// The root of a run whose gpu server is known by `service` (module docs, "The gpu
    /// session").
    #[must_use]
    pub const fn with_gpu_server(mut self, service: ServiceId) -> Self {
        self.gpu_server = Some(service);
        self
    }

    /// The number of the gpu server 497 hands sessions on, if the run has one.
    #[must_use]
    pub const fn gpu_server(&self) -> Option<ServiceId> {
        self.gpu_server
    }

    /// The root of a run whose syncpt server is known by `service` (module docs, "The family
    /// sessions").
    #[must_use]
    pub const fn with_syncpt_server(mut self, service: ServiceId) -> Self {
        self.syncpt_server = Some(service);
        self
    }

    /// The number of the syncpt server 202 hands sessions on, if the run has one.
    #[must_use]
    pub const fn syncpt_server(&self) -> Option<ServiceId> {
        self.syncpt_server
    }

    /// The root of a run whose nvsrv server is known by `service` (module docs, "The family
    /// sessions").
    #[must_use]
    pub const fn with_nvsrv_server(mut self, service: ServiceId) -> Self {
        self.nvsrv_server = Some(service);
        self
    }

    /// The number of the nvsrv server 1402 hands sessions on, if the run has one.
    #[must_use]
    pub const fn nvsrv_server(&self) -> Option<ServiceId> {
        self.nvsrv_server
    }

    /// The root of a run whose vi2 server is known by `service` (module docs, "The display
    /// session (305)").
    #[must_use]
    pub const fn with_vi2_server(mut self, service: ServiceId) -> Self {
        self.vi2_server = Some(service);
        self
    }

    /// The number of the vi2 server 305 hands sessions on, if the run has one.
    #[must_use]
    pub const fn vi2_server(&self) -> Option<ServiceId> {
        self.vi2_server
    }

    /// The root of a run whose video server is known by `service` (module docs, "Video (slice
    /// 15)").
    #[must_use]
    pub const fn with_video_server(mut self, service: ServiceId) -> Self {
        self.video_server = Some(service);
        self
    }

    /// The number of the video server 902, 906 and 1002 hand sessions on, if the run has one.
    #[must_use]
    pub const fn video_server(&self) -> Option<ServiceId> {
        self.video_server
    }

    /// Where the bootstrap is.
    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    /// What the last `Initialize` asked, if one was sent.
    #[must_use]
    pub const fn bootstrap(&self) -> Option<&BootstrapRequest> {
        self.bootstrap.as_ref()
    }

    /// How many bootstrap commands came in a phase other than the one PR requires (decision 7).
    /// It stops at `u32::MAX`.
    #[must_use]
    pub const fn out_of_order(&self) -> u32 {
        self.out_of_order
    }

    /// The mode the last `ConfigureNativeMemory` (115) named, if one was sent.
    #[must_use]
    pub const fn native_memory_mode(&self) -> Option<u64> {
        self.native_memory_mode
    }

    /// The events the 204 stub has made.
    #[must_use]
    pub const fn syncpoint_events(&self) -> SyncpointEvents {
        self.syncpoint_events
    }

    /// Moves the bootstrap to the phase `command` leads to, logging and counting it if the
    /// bootstrap was not in the phase PR requires (see "The bootstrap" in the module docs).
    fn step(&mut self, command: u32) {
        let Some((required, next)) = Phase::step(command) else {
            return;
        };
        if self.phase != required {
            self.out_of_order = self.out_of_order.saturating_add(1);
            tracing::warn!(
                command,
                phase = self.phase.word(),
                required = required.word(),
                "gsv: bootstrap command out of order; accepted (decision 7)"
            );
        }
        self.phase = next;
    }

    /// Logs a command PR runs only once the bootstrap is complete, if it is not.
    fn note_unready(&self, command: u32) {
        if self.phase != Phase::Ready {
            tracing::warn!(
                command,
                phase = self.phase.word(),
                "gsv: command before the bootstrap is complete; accepted (decision 7)"
            );
        }
    }
}

/// The root interface: what a session on the root's server is. It keeps nothing; the state is
/// the server's ([`RootState`]).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GsvRoot;

/// The session a stub hands out in place of a family's own (see "Stubs" in the module docs). It
/// has no commands.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Placeholder {
    command: u32,
}

impl Placeholder {
    /// The root command that handed it out.
    #[must_use]
    pub const fn command(&self) -> u32 {
        self.command
    }
}

commands! {
    impl GsvRoot {}
    tipc {
        INITIALIZE => initialize,
        BIND_PROCESS => bind_process,
        COMPLETE_BOOTSTRAP => complete_bootstrap,
        QUERY_MEMORY_MODE => query_memory_mode,
        CONFIGURE_NATIVE_MEMORY => configure_native_memory,
        INITIALIZE_SYNCPOINT_CONTEXT => stub,
        GET_SYNCPOINT_SESSION => get_syncpoint_session,
        BIND_SYNCPOINT_MEMORY => bind_syncpoint_memory,
        INITIALIZE_DISPLAY_CONTEXT => stub,
        FINALIZE_DISPLAY_CONTEXT => stub,
        OPEN_DISPLAY_SESSION => open_display_session,
        INITIALIZE_GPU_CONTEXT => stub,
        ACTIVATE_GPU_CONTEXT => stub,
        REGISTER_GPU_ADDRESS_SPACE => stub,
        OPEN_GPU_SESSION => open_gpu_session,
        GET_NVIDIA_DRIVER => get_nvidia_driver,
        CREATE_NATIVE_MEMORY => create_native_memory,
        DESTROY_NATIVE_MEMORY => destroy_native_memory,
        BIND_NATIVE_MEMORY => bind_native_memory,
        BIND_NATIVE_GPU_MEMORY => bind_native_gpu_memory,
        CREATE_NATIVE_IMAGE => create_native_image,
        DESTROY_NATIVE_IMAGE => destroy_native_image,
        CALCULATE_NATIVE_IMAGE_SIZE => calculate_image_requirements,
        CALCULATE_IMAGE_REQUIREMENTS => calculate_image_requirements,
        DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL => native_window_pool,
        CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL => native_window_pool,
        MAP_NATIVE_GPU_IMAGE => map_native_gpu_image,
        UNMAP_NATIVE_GPU_IMAGE => unmap_native_gpu_image,
        QUERY_NATIVE_GPU_IMAGE => query_native_gpu_image,
        INITIALIZE_VIDEO_DECODER => stub,
        OPEN_VIDEO_DECODER_CONTEXT_902 => open_video_context,
        FINALIZE_VIDEO_DECODER => stub,
        OPEN_VIDEO_DECODER_CONTEXT => open_video_context,
        CREATE_VIDEO_IMAGE_POOL => create_video_image_pool,
        DESTROY_VIDEO_IMAGE_POOL => destroy_video_image_pool,
        QUERY_VIDEO_IMAGE => query_video_image,
        CALCULATE_VIDEO_IMAGE_REQUIREMENTS => calculate_video_image_requirements,
        INITIALIZE_VIDEO_ENCODER => stub,
        OPEN_VIDEO_ENCODER_CONTEXT => open_video_context,
        FINALIZE_VIDEO_ENCODER => stub,
        INITIALIZE_FRAME_CONTEXT => stub,
        ACTIVATE_FRAME_CONTEXT => stub,
        SET_GRAPHICS_CLASS_1111 => stub,
        GET_NATIVE_EVENT_1121 => stub,
        SEQUENCE_COMMAND_1201 => stub,
        SEQUENCE_COMMAND_1202 => stub,
        SEQUENCE_COMMAND_1203 => stub,
        CREATE_CHAT_IMAGE_POOL => stub,
        CALCULATE_REQUIRED_BUFFER_MEMORY => stub,
        INITIALIZE_IMAGE_CONTEXT_1501 => stub,
        INITIALIZE_IMAGE_CONTEXT_1502 => stub,
        CREATE_CAMERA_CONTEXT => stub,
        OPEN_CAMERA_SESSION => open_camera_session,
    }
}

// No command: every request goes to the server's unknown-command policy (module docs, "Stubs").
commands! {
    impl Placeholder {}
}

impl GsvRoot {
    /// `Initialize` (4). in: copy handle 0, the process; `u64` applet resource user id, `u64`
    /// flags, four `u64` parameters. Everything is read before anything is recorded, so a short
    /// request changes nothing.
    fn initialize(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        // PR checks the process handle before the phase and the fields.
        req.copy_handle(0)?;
        let mut payload = req.payload();
        let applet_resource_user_id = payload.read_u64()?;
        let flags = payload.read_u64()?;
        let mut parameters = [0; bootstrap::PARAMETERS];
        for parameter in &mut parameters {
            *parameter = payload.read_u64()?;
        }
        let request = BootstrapRequest {
            applet_resource_user_id,
            flags,
            parameters,
        };
        if !request.is_local() {
            tracing::warn!(
                flags,
                "gsv: a remote-memory bootstrap; accepted as NX2 does, with no remote memory"
            );
        }
        let state = root_state(cx)?;
        state.step(INITIALIZE);
        state.bootstrap = Some(request);
        reply.ok()
    }

    /// `BindProcess` (11). in: copy handle 0, the process.
    fn bind_process(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        req.copy_handle(0)?;
        root_state(cx)?.step(BIND_PROCESS);
        reply.ok()
    }

    /// `CompleteBootstrap` (101).
    fn complete_bootstrap(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        root_state(cx)?.step(COMPLETE_BOOTSTRAP);
        reply.ok()
    }

    /// `QueryMemoryMode` (126). out: `u32` 0, no remote memory.
    fn query_memory_mode(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        root_state(cx)?.note_unready(QUERY_MEMORY_MODE);
        reply.payload().write_u32(bootstrap::NO_REMOTE_MEMORY_MODE);
        reply.ok()
    }

    /// `ConfigureNativeMemory` (115). in: `u64` mode. out: `u64` 0.
    fn configure_native_memory(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mode = req.payload().read_u64()?;
        let state = root_state(cx)?;
        state.note_unready(CONFIGURE_NATIVE_MEMORY);
        // PR `ConfigureNativeMemory`: `packet[2] == 0 || packet[2] == 2`, `packet[3] == 0`.
        if mode != 0 && mode != 2 {
            tracing::warn!(mode, "gsv: native memory mode PR refuses; accepted");
        }
        state.native_memory_mode = Some(mode);
        reply.payload().write_u64(bootstrap::NO_NATIVE_MEMORY_MODE);
        reply.ok()
    }

    /// `OpenGpuSession` (497). in: `u64` flags. out: copy handle, a session on the gpu server
    /// (module docs, "The gpu session"); the stub's placeholder in a run with none.
    fn open_gpu_session(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(gpu) = root_state(cx)?.gpu_server else {
            return Self::stub(this, cx, req, reply);
        };
        let flags = req.payload().read_u64()?;
        if flags != GPU_SESSION_FLAGS {
            tracing::warn!(
                flags,
                "gsv: 497 with flags PR refuses; accepted (decision 7)"
            );
        }
        hand_out(cx, reply, OPEN_GPU_SESSION, gpu)
    }

    /// `OpenDisplaySession` (305). in: `u64` flags. out: copy handle, a session on the vi2
    /// server (module docs, "The display session (305)"); the stub's placeholder in a run with
    /// none.
    fn open_display_session(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(vi2) = root_state(cx)?.vi2_server else {
            return Self::stub(this, cx, req, reply);
        };
        let flags = req.payload().read_u64()?;
        if !DISPLAY_SESSION_FLAGS.contains(&flags) {
            tracing::warn!(
                flags,
                "gsv: 305 with flags PR refuses; accepted (decision 7)"
            );
        }
        hand_out(cx, reply, OPEN_DISPLAY_SESSION, vi2)
    }

    /// `GetSyncpointSession` (202). in: nothing. out: copy handle, a session on the syncpt
    /// server (module docs, "The family sessions"); the stub's placeholder in a run with none.
    fn get_syncpoint_session(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        match root_state(cx)?.syncpt_server {
            Some(syncpt) => hand_out(cx, reply, GET_SYNCPOINT_SESSION, syncpt),
            None => Self::stub(this, cx, req, reply),
        }
    }

    /// `GetNvidiaDriver` (1402). in: nothing. out: copy handle, a session on the nvsrv server
    /// (module docs, "The family sessions"); the stub's placeholder in a run with none.
    fn get_nvidia_driver(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        match root_state(cx)?.nvsrv_server {
            Some(nvsrv) => hand_out(cx, reply, GET_NVIDIA_DRIVER, nvsrv),
            None => Self::stub(this, cx, req, reply),
        }
    }

    /// `BindSyncpointMemory` (204): binds the shared syncpoint page (module docs, "The shared
    /// syncpoint page (204)"); the stub's two events in a run without gsv's shared state.
    fn bind_syncpoint_memory(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(shared) = cx.family::<RootFamily>()?.shared.as_ref().map(Arc::clone) else {
            return Self::stub(this, cx, req, reply);
        };
        root_state(cx)?.note_unready(BIND_SYNCPOINT_MEMORY);
        page::bind(&shared, cx, req, reply)
    }

    /// A native memory or image command (module docs, "Native memory and images"): answered by
    /// `handle` from gsv's shared state, the command logged if the bootstrap is not complete;
    /// the stub's reply in a run without the shared state.
    fn native(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
        handle: fn(&Shared, &mut Context<'_>, &Request<'_>, &mut Reply) -> Outcome,
    ) -> Outcome {
        let Some(shared) = cx.family::<RootFamily>()?.shared.as_ref().map(Arc::clone) else {
            return Self::stub(this, cx, req, reply);
        };
        root_state(cx)?.note_unready(req.command_id());
        handle(&shared, cx, req, reply)
    }

    /// `CreateNativeMemory` (111): [`native::create_memory`].
    fn create_native_memory(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            native::create_memory(shared, req, reply)
        })
    }

    /// `DestroyNativeMemory` (112): [`native::destroy_memory`].
    fn destroy_native_memory(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, native::destroy_memory)
    }

    /// `BindNativeMemory` (114): [`native::bind_memory`].
    fn bind_native_memory(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, native::bind_memory)
    }

    /// `BindNativeGpuMemory` (423): [`native::bind_gpu_memory`].
    fn bind_native_gpu_memory(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, native::bind_gpu_memory)
    }

    /// `CreateNativeImage` (429): [`native::create_image`].
    fn create_native_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            native::create_image(shared, req, reply)
        })
    }

    /// `DestroyNativeImage` (430): [`native::destroy_image`].
    fn destroy_native_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, native::destroy_image)
    }

    /// `DestroyNativeWindowAliasImagePool` (314) and `CreateNativeWindowAliasImagePool` (315):
    /// [`native::window_pool`].
    fn native_window_pool(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            native::window_pool(shared, req, reply)
        })
    }

    /// `CalculateNativeImageSize` (435) and `CalculateImageRequirements` (1510):
    /// [`native::calculate_requirements`].
    fn calculate_image_requirements(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |_, _, req, reply| {
            native::calculate_requirements(req, reply)
        })
    }

    /// 902, 906 and 1002: a copied session on the video server (module docs, "Video (slice
    /// 15)"); the stub's placeholder in a run with none.
    fn open_video_context(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(video) = root_state(cx)?.video_server else {
            return Self::stub(this, cx, req, reply);
        };
        let command = req.command_id();
        if command == OPEN_VIDEO_DECODER_CONTEXT {
            let mut words = req.payload();
            let remote = low_byte(words.read_u32()?);
            let version = words.read_u64()?;
            if remote != 0 || version != VIDEO_LIBRARY_VERSION {
                tracing::warn!(
                    remote,
                    version,
                    "gsv: 906 with a remote-memory flag or library version PR refuses; accepted"
                );
            }
        }
        hand_out(cx, reply, command, video)
    }

    /// 602 `OpenCameraSession`: in a `u64` device type; out a copied camera session
    /// ([`crate::video::camera`]); the stub's placeholder in a run without gsv's shared state.
    fn open_camera_session(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if !cx.family::<RootFamily>()?.has_page() {
            return Self::stub(this, cx, req, reply);
        }
        let device = req.payload().read_u64()?;
        root_state(cx)?.note_unready(OPEN_CAMERA_SESSION);
        if device > camera::MAX_DEVICE_TYPE {
            return Err(camera::GSV_UNKNOWN_ARGUMENT.into());
        }
        reply.copy_session(Box::new(CameraSession));
        reply.ok()
    }

    /// 991 `CalculateVideoImageRequirements`: [`pools::calculate_requirements`].
    fn calculate_video_image_requirements(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |_, _, req, reply| {
            pools::calculate_requirements(req, reply)
        })
    }

    /// 911 `CreateVideoImagePool`: [`pools::create_pool`].
    fn create_video_image_pool(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            pools::create_pool(shared, req, reply)
        })
    }

    /// 912 `DestroyVideoImagePool`: [`pools::destroy_pool`].
    fn destroy_video_image_pool(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, pools::destroy_pool)
    }

    /// 961 `QueryVideoImage`: [`pools::query_video_image`].
    fn query_video_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            pools::query_video_image(shared, req, reply)
        })
    }

    /// 426 `MapNativeGpuImage`: [`pools::map_gpu_image`].
    fn map_native_gpu_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            pools::map_gpu_image(shared, req, reply)
        })
    }

    /// 427 `UnmapNativeGpuImage`: [`pools::unmap_gpu_image`].
    fn unmap_native_gpu_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            pools::unmap_gpu_image(shared, req, reply)
        })
    }

    /// 428 `QueryNativeGpuImage`: [`pools::query_gpu_image`].
    fn query_native_gpu_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::native(this, cx, req, reply, |shared, _, req, reply| {
            pools::query_gpu_image(shared, req, reply)
        })
    }

    /// Every other root command: logs it and replies success with its known reply shape (module
    /// docs, "Stubs").
    fn stub(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let id = req.command_id();
        let Some(row) = root_command(id) else {
            // The command table and `ROOT_COMMANDS` list the same ids (a test checks): a
            // mistake in the service, not something a guest did.
            tracing::error!(command = id, "gsv: a stub with no row in the command table");
            return Err(NOT_SUPPORTED.into());
        };
        let phase = root_state(cx)?.phase;
        tracing::info!(
            command = id,
            name = row.name,
            mark = row.mark.word(),
            phase = phase.word(),
            "gsv: stub"
        );
        // Only 204 returns events: the two it keeps.
        let events = if row.reply.handles.contains(&Handed::Event) {
            syncpoint_events(cx)?.to_vec()
        } else {
            Vec::new()
        };
        let mut next_event = events.into_iter();
        for handed in row.reply.handles {
            match handed {
                Handed::Session => {
                    reply.copy_session(Box::new(Placeholder { command: id }));
                }
                Handed::Event => {
                    let event = next_event.next().ok_or(Fault::from(NOT_SUPPORTED))?;
                    reply.copy_handle(event);
                }
            }
        }
        for _ in 0..row.reply.words {
            reply.payload().write_u32(0);
        }
        reply.ok()
    }
}

/// Answers `command` with a copied session on the server known by `service`, logging the
/// command if the bootstrap is not complete (module docs, "The family sessions").
fn hand_out(cx: &mut Context<'_>, reply: &mut Reply, command: u32, service: ServiceId) -> Outcome {
    root_state(cx)?.note_unready(command);
    reply.copy_service(service);
    reply.ok()
}

/// The root's state, from the server's family.
fn root_state<'c>(cx: &'c mut Context<'_>) -> Result<&'c mut RootState, Fault> {
    Ok(&mut cx.family::<RootFamily>()?.state)
}

/// The two events the 204 stub hands out, made the first time and kept (module docs, "Stubs").
fn syncpoint_events(cx: &mut Context<'_>) -> Result<[ObjectId; 2], Fault> {
    let first = match root_state(cx)?.syncpoint_events {
        SyncpointEvents::Both(first, second) => return Ok([first, second]),
        SyncpointEvents::One(first) => first,
        SyncpointEvents::None => {
            let first = cx.create_event()?;
            root_state(cx)?.syncpoint_events = SyncpointEvents::One(first);
            first
        }
    };
    let second = cx.create_event()?;
    root_state(cx)?.syncpoint_events = SyncpointEvents::Both(first, second);
    Ok([first, second])
}

/// The interfaces a root server's sessions can be.
#[must_use]
pub fn root_interfaces() -> Interfaces {
    Interfaces::new()
        .with::<GsvRoot>()
        .with::<Placeholder>()
        .with::<CameraSession>()
}

/// The library version 906 carries that PR takes (`DecodeVideoContext`:
/// `request->library_version == 0x300030000ULL`; NX2 row 146: "title 4 sends version
/// 0x3'00030000"). Another is logged and accepted (module docs, "Video (slice 15)").
pub const VIDEO_LIBRARY_VERSION: u64 = 0x3_0003_0000;

/// The flags word 497 carries: PR's `OpenGpuSession` requires `flags == 4` (graphics-server,
/// "The root": "497 takes a u64 (4)").
pub const GPU_SESSION_FLAGS: u64 = 4;

/// The flags words 305 carries: PR's `IsSupportedDisplaySession` (`flags == 0x300020000ULL ||
/// flags == 0x300030000ULL || flags == 0x300040000ULL`; graphics-server, "The root").
pub const DISPLAY_SESSION_FLAGS: [u64; 3] = [0x3_0002_0000, 0x3_0003_0000, 0x3_0004_0000];

/// The root's server: every session a [`GsvRoot`], sharing one [`RootState`], in a run with no
/// gpu server.
#[must_use]
pub fn root_server(policy: UnknownPolicy) -> Server {
    root_server_for(RootState::new(), policy)
}

/// The root's server over `state`: [`RootState::new`], and in a run with a gpu server
/// [`RootState::with_gpu_server`]. A snapshot whose gpu server number differs from `state`'s is
/// refused (module docs, "The gpu session"). 204 answers with the stub's two events.
#[must_use]
pub fn root_server_for(state: RootState, policy: UnknownPolicy) -> Server {
    root_server_over(
        RootFamily {
            state,
            shared: None,
        },
        policy,
    )
}

/// The root's server over `state`, as [`root_server_for`], whose 204 binds the shared syncpoint
/// page in gsv's `shared` state (module docs, "The shared syncpoint page (204)").
#[must_use]
pub fn root_server_with_page(
    state: RootState,
    policy: UnknownPolicy,
    shared: Arc<Shared>,
) -> Server {
    root_server_over(
        RootFamily {
            state,
            shared: Some(shared),
        },
        policy,
    )
}

/// The root's server with `family`.
fn root_server_over(family: RootFamily, policy: UnknownPolicy) -> Server {
    Server::new(ROOT_NAME, policy)
        .with_root(|| Box::new(GsvRoot))
        .with_interfaces(root_interfaces())
        .with_family(family)
        .with_family_objects(syncpoint_event_objects)
}

/// The restore check of the kernel objects a [`RootState`] keeps (module docs, "Restoring"):
/// each of 204's events it keeps is a live readable event the root's server made, as
/// `syncpoint_events` makes them, the first before the second.
fn syncpoint_event_objects(
    saved: &RootState,
    kernel: &dyn RestoredKernel,
    service: ServiceId,
) -> Result<(), &'static str> {
    let (first, second) = match saved.syncpoint_events {
        SyncpointEvents::None => (None, None),
        SyncpointEvents::One(first) => (Some(first), None),
        SyncpointEvents::Both(first, second) => (Some(first), Some(second)),
    };
    let kept = [
        (
            first,
            "its root's first syncpoint event is not a live readable event its server made",
        ),
        (
            second,
            "its root's second syncpoint event is not a live readable event its server made",
        ),
    ];
    for (event, refusal) in kept {
        if let Some(event) = event
            && !kernel.is_service_readable_event(service, event)
        {
            return Err(refusal);
        }
    }
    Ok(())
}

/// What every root session shares (the root server's family): the [`RootState`], and in a run
/// with the shared syncpoint page the handle to gsv's shared state (module docs, "The shared
/// syncpoint page (204)").
///
/// It implements [`Family`] by hand: a snapshot holds the [`RootState`] alone, as the
/// [`RootState`]'s own `serde` form, and loading one keeps the handle. Loading refuses a
/// snapshot whose gpu, syncpt, nvsrv or vi2 server number is not the run's
/// ([`SnapshotError::FamilyWiring`]), in that order, before it changes anything (module docs,
/// "The family sessions"). The root's server checks 204's events against the kernel before
/// loading begins (module docs, "Restoring").
pub struct RootFamily {
    state: RootState,
    shared: Option<Arc<Shared>>,
}

impl RootFamily {
    /// The root's state.
    #[must_use]
    pub const fn state(&self) -> &RootState {
        &self.state
    }

    /// Whether 204 binds the shared syncpoint page: the root was built with gsv's shared state.
    #[must_use]
    pub const fn has_page(&self) -> bool {
        self.shared.is_some()
    }
}

impl Family for RootFamily {
    /// The [`RootState`]'s own bytes.
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        self.state.save_state()
    }

    /// Reads a [`RootState`] and, if its gpu, syncpt, nvsrv and vi2 server numbers are the run's,
    /// takes it; the handle to gsv's shared state stays.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let mut saved = RootState::new();
        saved.load_state(bytes)?;
        if saved.gpu_server != self.state.gpu_server {
            return Err(SnapshotError::FamilyWiring(
                "its root names another gpu server",
            ));
        }
        if saved.syncpt_server != self.state.syncpt_server {
            return Err(SnapshotError::FamilyWiring(
                "its root names another syncpt server",
            ));
        }
        if saved.nvsrv_server != self.state.nvsrv_server {
            return Err(SnapshotError::FamilyWiring(
                "its root names another nvsrv server",
            ));
        }
        if saved.vi2_server != self.state.vi2_server {
            return Err(SnapshotError::FamilyWiring(
                "its root names another vi2 server",
            ));
        }
        if saved.video_server != self.state.video_server {
            return Err(SnapshotError::FamilyWiring(
                "its root names another video server",
            ));
        }
        self.state = saved;
        Ok(())
    }

    /// When the server's last session has closed (`open` is 0) in a run with the shared page,
    /// queues [`Completion::RootClosed`] for the timer thread's drain, which lets the page and
    /// its two transfer memories go if no root session is open by then (module docs, "The
    /// shared syncpoint page (204)"). Called with the kernel's lock held when a process's end
    /// closed the session: it only pushes, which takes only the queue's lock
    /// ([`crate::completion`], "Never blocking"), never gsv's shared lock or the kernel's.
    fn session_closed(&mut self, open: usize) {
        if open != 0 {
            return;
        }
        if let Some(shared) = &self.shared {
            shared.queue().push(Completion::RootClosed);
        }
    }
}
