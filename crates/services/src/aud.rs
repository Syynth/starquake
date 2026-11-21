//! `aud:u`, the Switch 2 audio entry point, as far as a title's start-up asks: command 2 hands
//! out an audio renderer manager, whose `GetWorkBufferSize` answers as Eden's does.
//!
//! Ported from, paths relative to the reference folder (all GPL-3.0-or-later):
//!
//! - `eden-nx2/src/core/hle/service/audio/nx2_audio_user_client.h` @ 50fa86bb
//!   (`INx2AudioUserClient`): `aud:u`'s command 2 ([`IAudioUserClient`]);
//! - `eden-nx2/src/core/hle/service/audio/audio.cpp` @ 50fa86bb (`LoopProcess`:
//!   `RegisterNamedService("aud:u", ..)`): the name ([`U_NAME`]);
//! - `eden-nx2/src/core/hle/service/audio/audio_renderer_manager.cpp` @ 50fa86bb
//!   (`IAudioRendererManager`): the manager's command table and `GetWorkBufferSize`
//!   ([`IAudioRendererManager`]), byte for byte the same in Eden PR #4588 @ e5fb3b97;
//! - `eden-nx2/src/core/hle/service/audio/errors.h` @ 50fa86bb: [`INVALID_REVISION`] and
//!   [`INVALID_ARGUMENT`];
//! - the size itself: [`work_buffer`], which lists its own sources.
//!
//! Row 38 of Eden-NX2's blocker log (`eden-nx2/NX2_NOTES.md`; cited by
//! `docs/switch2-eden-diff-scoping.md`, "Other") records the fault: a title waits for `aud:u`,
//! the Switch 2 audio entry (`nn::audio::CreateUserClientSessionByHipc`), whose command 2
//! returns an `IAudioRendererManager`; Eden-NX2's fix is `INx2AudioUserClient`, backed by
//! Eden's `audren:u` manager. Its header says the title uses the object "exactly like
//! IAudioRendererManager (vtable slot 1 = GetWorkBufferSize with a Switch 1 parameter block)":
//! command 1 is the one the title calls next, and the one ported here.
//!
//! # Commands
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `aud:u` ([`IAudioUserClient`]) | 2 | `GetAudioRendererManager` | a `u32` (not read) | an [`IAudioRendererManager`] |
//! | [`IAudioRendererManager`] | 1 | `GetWorkBufferSize` | a [`RendererParameter`] (0x34 bytes) | `u64` size; [`INVALID_REVISION`], or [`INVALID_ARGUMENT`] (a guess) |
//!
//! `GetWorkBufferSize` answers [`work_buffer_size`] of the parameter: [`INVALID_REVISION`] for
//! a revision past 15 (Eden's `Manager::GetWorkBufferSize`: `if
//! (!CheckValidRevision(params.revision)) return Service::Audio::ResultInvalidRevision`), and
//! [`INVALID_ARGUMENT`] for a parameter whose size would wrap Eden's arithmetic or that has a
//! negative splitter destination count where a term reads it ([`work_buffer`], "Where this
//! differs from Eden"). The revision is checked first.
//!
//! # The two references
//!
//! Conflict (graphics-server decision 8's rule, `docs/specs/graphics-server.md`): Eden-NX2's
//! `aud:u` has only command 2 (named `GetAudioRendererManager (guessed)`), which pops a `u32`
//! it only logs and makes a new manager on each call. Eden PR #4588's (`INativeAudioManager`,
//! `eden-pr4588/src/core/hle/service/audio/audio.cpp` @ e5fb3b97) has 0
//! (`GetAudioInManager`), 1 (`GetAudioOutManager`) and 2 (`GetAudioRendererManager`), reads
//! nothing for 2, hands out the one manager `audren:u` serves too, and answers any TIPC request
//! with `Kernel::ResultNotImplemented`. Both agree on command 2 returning an
//! `IAudioRendererManager`, which is what is served. The rest:
//!
//! - **The `u32` is not read**, so a request with it (Eden-NX2's) or without it (PR #4588's)
//!   is served alike (`get_audio_renderer_manager_reads_no_word`).
//! - **A new manager per call**, as Eden-NX2 makes. It keeps nothing, and its one ported
//!   command reads nothing but the request, so one manager or many answer the same. When
//!   `OpenAudioRenderer` is ported, Eden's renderer-session count lives in the manager (Eden's
//!   `impl`, `AudioCore::Renderer::Manager`, two sessions at most); whether it is one count per
//!   server (PR #4588's shared manager) or per manager (Eden-NX2's) is to be settled then. Per
//!   server, it would be the server's family state, which this design leaves room for.
//! - **Commands 0 and 1 are left out**: PR #4588's audio in and out managers, which no row
//!   records a title asking for and Eden-NX2 does not serve. They go to the server's
//!   unknown-command policy.
//! - **A TIPC request** goes to the unknown-command policy (no interface here has a TIPC
//!   table): auto-stubbed in a run, where PR #4588 answers `ResultNotImplemented`
//!   (`a_tipc_request_goes_to_the_unknown_command_policy`).
//!
//! # Left out
//!
//! Every other command goes to the server's unknown-command policy, which under auto-stub logs
//! it once, answers success with nothing, and lists it among the run's stubbed commands:
//!
//! - `aud:u`: 0 and 1 (PR #4588's, above).
//! - [`IAudioRendererManager`]: `OpenAudioRenderer` (0), `GetAudioDeviceService` (2),
//!   `OpenAudioRendererForManualExecution` (3, which Eden lists without a handler) and
//!   `GetAudioDeviceServiceWithRevisionInfo` (4). Commands 0, 2 and 4 return an object in
//!   Eden (an `IAudioRenderer`, an `IAudioDevice`), which an auto-stubbed reply does not hold;
//!   they are the next to port when a title's run lists them (row 39 records a title's
//!   `RequestUpdate` on an opened renderer). Audio output itself is not designed yet: there is
//!   no host audio, and nothing here rules one out, since the renderer would be a session this
//!   manager hands out.
//!
//! # Snapshots
//!
//! Neither interface keeps state, and the server has no family state, so a snapshot holds the
//! sessions' interfaces only.

use serde::{Deserialize, Serialize};
use starquake_kernel::result::ResultCode;

use crate::commands;
use crate::framework::result::code;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

pub mod work_buffer;

pub use work_buffer::{RendererParameter, WorkBufferError, work_buffer_size};

#[cfg(test)]
mod tests;

/// The name `aud:u` is registered under (`eden-nx2/src/core/hle/service/audio/audio.cpp`,
/// `LoopProcess`: `RegisterNamedService("aud:u", std::make_shared<INx2AudioUserClient>(system),
/// 30)`).
pub const U_NAME: &str = "aud:u";

/// `ErrorModule::Audio` (`eden-nx2/src/core/hle/result.h` @ 50fa86bb: `Audio = 153`).
pub const AUDIO_MODULE: u32 = 153;

/// `ResultInvalidRevision` (`errors.h`: `constexpr Result ResultInvalidRevision{ErrorModule::Audio,
/// 1537}`), which `GetWorkBufferSize` answers for a revision past 15.
pub const INVALID_REVISION: ResultCode = code(AUDIO_MODULE, 1537);

/// `ResultInvalidArgument` (`errors.h`: `constexpr Result ResultInvalidArgument{ErrorModule::Audio,
/// 900}`), which `GetWorkBufferSize` answers for a parameter whose size Eden's arithmetic would
/// wrap, or with a negative splitter destination count where a term reads it. A guess: Eden has
/// no such check and answers the wrapped size ([`work_buffer`], "Where this differs from
/// Eden").
pub const INVALID_ARGUMENT: ResultCode = code(AUDIO_MODULE, 900);

/// The root interface of an `aud:u` session: Eden-NX2's `INx2AudioUserClient`, PR #4588's
/// `INativeAudioManager` (module docs, "The two references"). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IAudioUserClient;

/// An audio renderer manager (`IAudioRendererManager`): what `aud:u`'s command 2 hands out. Its
/// one ported command reads only the request, so it keeps nothing (module docs).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IAudioRendererManager;

/// The ids of the commands ported here, per interface (the references' tables).
pub mod command {
    /// `aud:u`'s commands.
    pub mod user_client {
        /// `GetAudioRendererManager` (Eden-NX2: `{2,
        /// &INx2AudioUserClient::GetAudioRendererManager, "GetAudioRendererManager
        /// (guessed)"}`; PR #4588: `{2, &INativeAudioManager::GetAudioRendererManager,
        /// "GetAudioRendererManager"}`).
        pub const GET_AUDIO_RENDERER_MANAGER: u32 = 2;
    }

    /// `IAudioRendererManager`'s commands.
    pub mod renderer_manager {
        /// `GetWorkBufferSize` (`{1, D<&IAudioRendererManager::GetWorkBufferSize>,
        /// "GetWorkBufferSize"}`).
        pub const GET_WORK_BUFFER_SIZE: u32 = 1;
    }
}

commands! {
    impl IAudioUserClient {
        command::user_client::GET_AUDIO_RENDERER_MANAGER => get_audio_renderer_manager,
    }
}

commands! {
    impl IAudioRendererManager {
        command::renderer_manager::GET_WORK_BUFFER_SIZE => get_work_buffer_size,
    }
}

impl IAudioUserClient {
    /// `GetAudioRendererManager`. in: a `u32` Eden-NX2 pops and only logs (not read). out: a
    /// new [`IAudioRendererManager`].
    fn get_audio_renderer_manager(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IAudioRendererManager));
        reply.ok()
    }
}

impl IAudioRendererManager {
    /// `GetWorkBufferSize`. in: a [`RendererParameter`]. out: the `u64` [`work_buffer_size`];
    /// [`INVALID_REVISION`] or [`INVALID_ARGUMENT`] if it has none (module docs, "Commands").
    fn get_work_buffer_size(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let params = RendererParameter::read(&mut req.payload())?;
        match work_buffer_size(&params) {
            Ok(size) => {
                tracing::debug!(revision = params.revision, size, "aud: GetWorkBufferSize");
                reply.payload().write_u64(size);
                reply.ok()
            }
            Err(error) => {
                tracing::debug!(%error, "aud: GetWorkBufferSize refused");
                Err(result_of(error).into())
            }
        }
    }
}

/// The result `GetWorkBufferSize` answers for `error` (module docs, "Commands").
const fn result_of(error: WorkBufferError) -> ResultCode {
    match error {
        WorkBufferError::InvalidRevision { .. } => INVALID_REVISION,
        WorkBufferError::Overflow | WorkBufferError::NegativeSplitterDestinations(_) => {
            INVALID_ARGUMENT
        }
    }
}

/// The interfaces an `aud:u` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IAudioUserClient>()
        .with::<IAudioRendererManager>()
}

/// The `aud:u` server.
#[must_use]
pub fn u_server(policy: UnknownPolicy) -> Server {
    Server::new(U_NAME, policy)
        .with_root(|| Box::new(IAudioUserClient))
        .with_interfaces(interfaces())
}
