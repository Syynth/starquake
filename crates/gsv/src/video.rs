//! The video codec sessions: the video context the root's 902, 906 and 1002 hand out, and the
//! decoder service it makes, with a null decoder (`docs/specs/graphics-server.md`, "Video codec
//! sessions"; port slice 15; decision 10: "no decoding in the first slices").
//!
//! * [`layout`]: the layout of a video image pool (root 991, 911, 961, 428).
//! * [`camera`]: the camera session root 602 hands out, with PR's reply shapes.
//!
//! The pools themselves, and the root commands that make and describe them, are in the native
//! tables ([`crate::native::pools`]).
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`INx2VideoSession`), and its blocker log `eden-nx2/NX2_NOTES.md` @ 50fa86bb, whose rows
//!   "row N" names (rows 105, 146 and 148 are the video rows, observed).
//! - **PR:** Eden PR #4588,
//!   `eden-pr4588/src/core/hle/service/am/service/native_video_decoder.cpp` @ e5fb3b97
//!   (`INativeVideoContext`, `INativeVideoCodec`) and `graphics_supervisor_protocol.h` @
//!   e5fb3b97 (`DecodeVideoImageRegistration`, `DecodeVideoDecoderConfiguration`,
//!   `DecodeVideoDecoderInitialization`, `DecodeVideoDecoderInitialization200`).
//!
//! # The server
//!
//! [`video_server`] is the video family's server (decision 6: one server per family). Its root
//! interface is the [`VideoContext`]: the root hands sessions on it out with 902, 906 and 1002
//! once it is given the server's number ([`crate::root::RootState::with_video_server`]), which
//! the wiring (`starquake-core`'s `hle::graphics`) does in a run. A context's TIPC 1 makes a
//! [`DecoderService`] on the same server. The family state ([`VideoFamily`]) is the handle to
//! gsv's shared state, where the pools are; it saves nothing.
//!
//! **conflict: 1002.** NX2 reads it as a second decoder (`OpenVideoDecoderSession2`), PR as the
//! video encoder (`OpenVideoEncoderContext`, whose codec refuses to initialise as a decoder).
//! The reply is a session either way (row 105, observed). By decision 8, the larger answer: a
//! context like 902's, whose decoder service works
//! (`the_root_hands_out_video_sessions_that_count_frames`, in `starquake-core`'s
//! `hle/tests/graphics/video.rs`). Verification 1 settles it.
//!
//! # The context
//!
//! | Id | Command | In | Out | Source |
//! |---|---|---|---|---|
//! | 1 | [`CREATE_DECODER`] | nothing | copy handle: a new [`DecoderService`] | row 105; PR `CreateDecoder` (`0x11, 0`); NX2 `CreateVideoDecoderService` ("handle at word 3, result at word 4") |
//!
//! # The decoder service
//!
//! Every command is TIPC. Words are counted from the start of the message as PR counts them:
//! the two header words, then the A descriptor's three words where the command has one (PR's
//! headers `0x100075` and `0x10007d` name one A descriptor; NX2: "Words 2-4 are the A
//! descriptor"), then the data.
//!
//! | Id | Command | Words | Out | Source |
//! |---|---|---|---|---|
//! | 3 | [`GET_SUBMISSION_EVENT`] | nothing | copy handle: the "ready for input" event | PR `GetSubmissionEvent` (`0x13, 0`); NX2 `GetEvent` |
//! | 6 | [`START`] | nothing | nothing | PR (`0x16, 0`); NX2 `Start` |
//! | 8 | [`FLUSH`] | nothing | nothing | PR (`0x18, 0`); NX2 `Flush` |
//! | 101 | [`CONFIGURE`] | A: the codec's name; width 5, height 6, frame rate 7, input capacity 8, four words 9–12 | `u64` 0 | PR `DecodeVideoDecoderConfiguration` (`0x100075, 8`); NX2 `Configure` |
//! | 109 | [`SUBMIT`] | A: one compressed frame; timestamp (µs) 5–6, end of stream 7 (low byte) | `u64` the frame's bytes | PR `Submit` (`0x10007d, 3`); NX2 `SubmitInput` |
//! | 115 | [`FINALIZE`] | nothing | nothing | PR (`0x83, 0`); NX2 `Finalize` |
//! | 200 | [`INITIALIZE_200`] | copy handle 3: work buffer; size 4–5, codec 6, input format 7, output format 8 | nothing | PR `DecodeVideoDecoderInitialization200` (`0xd8, 0x80000005`); NX2 `Initialize` |
//! | 201 | [`INITIALIZE`] | as 200, then output mode 9 | nothing | PR `DecodeVideoDecoderInitialization` (`0xd9, 0x80000006`) |
//! | 207 | [`STOP`] | nothing | nothing | PR (`0xdf, 0`); NX2 `Stop` |
//! | 212 | [`UNREGISTER_IMAGE`] | image 2 | nothing | PR (`0xe4, 8`); NX2 `UnregisterBuffer` |
//! | 213 | [`GET_OUTPUT_PROPERTIES`] | nothing | sixteen words: image, configured width, height, frame width, height, 0, timestamp (ns) as a `u64`, end of stream, zeros | row 148; PR `GetOutputProperties` (`0xe5, 0`); NX2 `GetOutputInfo` |
//! | 311 | [`REGISTER_IMAGE`] | image 2 | a native sync handle: `SyncCounterId`, mode 0, `u64` value, then two `u64` zeros | row 148; PR `DecodeVideoImageRegistration` (`0x147, 4`); NX2 `RegisterBuffer` |
//! | 314 | [`RELEASE_IMAGE`] | image 2 (NX2: a release fence at 4–7, not read) | as 311 | row 148; PR (`0x14a, 8`); NX2 `ReleaseBuffer` |
//!
//! A request that lacks a data word its command reads is the framework's
//! `INVALID_CMIF_REQUEST`; one with more words is read. A TIPC command not in the table, and a
//! CMIF request, go to the server's unknown-command policy.
//!
//! # The null decoder
//!
//! Nothing is decoded (decision 10): each submitted frame is finished at once into the next
//! output image handed out, with no picture (the image keeps its contents), as NX2 did before
//! row 148 ("the movie plays with no picture"). What that keeps:
//!
//! - **Handouts.** 311 and 314 each hand out an output image with the next value of the
//!   decoder's syncpoint: the `n`th handout since the syncpoint was allocated carries `base + n`
//!   (wrapping), `base` being the counter when 311 allocated it (NX2 `HandOut`: `m_sync_base +
//!   ++m_handed`; PR `++next_completion_value` from `ReadAllocatedSyncpoint`). The handouts wait
//!   in a queue, oldest first (NX2 `m_free`: "Output images, in the order the client acquires
//!   them").
//! - **Frames.** A submitted frame waits in the input queue until a handout is waiting; then the
//!   oldest frame takes the oldest handout, its output goes in the output queue for 213, and the
//!   syncpoint is brought to that handout's value ([`DecoderService::handed`]). So the handles
//!   count frames in order: the `n`th handout's value is reached when the `n`th frame is
//!   finished, and not before (`sync_handles_count_frames_in_order`). A handout 212 takes out
//!   leaves a gap: the next frame's value passes it, so every later value is still reached by
//!   its own frame (`an_unregistered_image_s_handouts_are_skipped`).
//! - **The ready event.** It is made by the first 3 or 101, and is signalled whenever the input
//!   queue has room ([`MAX_PENDING`]) after a command, and cleared when it is full.
//! - **Flush and stop (8, 207).** The frames waiting for an image are dropped; the handouts keep
//!   their place and values, and finished outputs stay for 213 (NX2 `Flush`: "Images already
//!   handed out keep their place, so the sync values stay in step with the client's ring";
//!   `flush_keeps_images_already_handed_out`).
//! - **Finalize (115).** Everything is forgotten (PR `Reset`): the syncpoint is freed, and the
//!   configuration, images, handouts, frames and outputs are dropped; the event is kept,
//!   cleared.
//!
//! # Bounds
//!
//! What a title can make a decoder service hold: [`MAX_REGISTERED`] images (PR: `registered_images.size()
//! < 8`, `OutOfResource`), [`MAX_HANDOUTS`] waiting handouts, [`MAX_PENDING`] waiting frames
//! (PR `MaximumPendingInputs`, answered [`VIDEO_QUEUE_FULL`]) and [`MAX_OUTPUTS`] finished
//! outputs (a frame waits while the output queue is full), one syncpoint and one event. A
//! command's work is bounded by those. A session the title closes keeps its syncpoint and its
//! event: the service framework tells a session nothing when it closes, as for the syncpt
//! sessions (`crate::syncpt`, "Where this differs from the references"); a title that keeps
//! opening decoder services, registering an image and closing them uses up the syncpoints. A
//! known gap, left for a follow-up.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `video/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | Commands out of PR's order (a submit before 6, a register after 6, a configure twice, a start with no image, and so on) | `InvalidState` | accepted | accepted, as NX2 and as the root's commands (decision 7) (`commands_are_taken_in_any_order`) |
//! | 200 and 201: the codec, the formats and the work buffer | checked (`NotImplemented`, `InvalidHandle`, `InvalidSize`) | not read | read and logged, not checked: nothing is decoded (`initialize_reads_its_words_and_checks_nothing`) |
//! | 101's words | checked (`InvalidCombination`); the codec must be AV1 or H.264 | not checked; VP9 taken | not checked; any codec taken (`configure_takes_any_codec_and_size`) |
//! | 101's reply | `u64` 0 | nothing | **conflict**: PR's, the larger (`configure_takes_any_codec_and_size`) |
//! | 109's reply | `u64` the frame's bytes (`consumed_size`) | `u64` a submission count | **conflict**: PR's: neither contains the other, both ran, and PR is the reference that reads the module (decision 8's last clause) (`a_submit_answers_the_frame_s_bytes`) |
//! | 109's end-of-stream byte above 1, or a frame whose emptiness disagrees with it, or larger than the configured capacity | `InvalidCombination`, `InvalidSize` | taken; any byte not 0 is the end | as NX2 (`any_end_byte_but_0_ends_the_stream`) |
//! | 109 when the input queue is full | [`VIDEO_QUEUE_FULL`] after [`MAX_PENDING`] | queued, unbounded | PR's (`a_full_input_queue_refuses_the_next_frame`) |
//! | 311 and 314's reply | eight words: the handle, then two `u64` zeros | four words: the handle | **conflict**: PR's, the larger (`register_answers_a_sync_handle`) |
//! | The handle's value | a `u64` that passes 2^32 | a `u32`, wrapping | NX2's: the counter is 32 bits and mode 0 compares 32-bit wrapping; the high word is 0 (`handle_values_wrap_at_32_bits`) |
//! | 311 of an image no pool holds, or registered already | `NotFound`, `InvalidState` | taken | PR's (`register_checks_in_pr_s_order`) |
//! | 311 of an image smaller than the configuration | `InvalidSize` | taken | taken: nothing is written into it |
//! | 314 of an image waiting already | a fresh value, no new handout: the old value is never reached in order | a second handout | NX2's: every handout is a frame's (`releasing_a_waiting_image_hands_it_out_again`) |
//! | 314's release fence | must be zero (`NotImplemented`) | waited for before writing the image | not read: nothing writes the image |
//! | Waiting handouts | at most [`MAX_REGISTERED`] images, but 314 re-tickets without bound | unbounded | at most [`MAX_HANDOUTS`], then `OutOfResource` (`handouts_are_bounded_at_the_limit`) |
//! | Finished outputs not yet read | unbounded | unbounded | at most [`MAX_OUTPUTS`]; a frame then waits (`outputs_are_bounded_and_a_frame_waits`) |
//! | 212 of an image not registered, or while started | `NotFound`, `InvalidState` | taken | `NotFound`; any time (`unregister_refuses_an_unknown_image`) |
//! | 8 (flush) | AV1: `NotImplemented`; H.264: decodes and drains what waits | drops what waits | NX2's (`flush_keeps_images_already_handed_out`) |
//! | 207 (stop) | drops the waiting frames and the finished outputs, and clears started | as its flush | NX2's: as 8 (`stop_is_flush`) |
//! | The ready event's signals | signalled when the queue has room after a pump, from 101 on | signalled on every 3, and every 4 ms while there is room | signalled after every command that leaves room, from the first 3 or 101: no timer (`the_ready_event_follows_the_input_queue`) |
//! | A frame's timestamp in ns | `timestamp * 1000`, `u64` wrapping | the same | the same (`the_timestamp_is_in_ns_wrapping`) |
//!
//! # What is recorded and what is saved
//!
//! Every request is one IPC event, recorded by the service framework. The null decoder finishes
//! frames inside the request that lets them finish, so a replay of the requests finishes them
//! again: the spec's `VideoFrameDone` note is for the decoding worker (decision 10's later
//! slice), whose frames finish on a host thread, and is not written here
//! (`crate::note`: code 7, "reserved for the video worker"). The syncpoint increments push
//! `SyncpointChanged` for a watched id, which the drain records as any other. A session is plain
//! data saved with the server, checked as it is read back ([`DecoderServiceError`]). Not
//! checked there: that the syncpoint is allocated in the restored `Syncpoints` and its counter
//! is `base` plus the last finished handout's number, and that the event is one the kernel
//! holds; a whole-machine restore checks those across the pieces.

pub mod camera;
pub mod layout;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::SyncpointId;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, Outcome, Reply, Request, Server, SnapshotError,
    UnknownPolicy,
};
use thiserror::Error;

use crate::gpu::low_byte;
use crate::shared::Shared;
use command::{
    CONFIGURE, CREATE_DECODER, FINALIZE, FLUSH, GET_OUTPUT_PROPERTIES, GET_SUBMISSION_EVENT,
    INITIALIZE, INITIALIZE_200, REGISTER_IMAGE, RELEASE_IMAGE, START, STOP, SUBMIT,
    UNREGISTER_IMAGE,
};

/// The video server's name. Starquake's own: on a console the codec sessions come from the
/// graphics server's root, and the family has no registered name in any reference.
pub const VIDEO_NAME: &str = "gsv:video";

/// The TIPC command ids of the video context and the decoder service, each with its source.
pub mod command {
    /// The context's `CreateDecoder` (PR `{1, &INativeVideoContext::CreateDecoder}`; NX2
    /// `{1, .., "CreateVideoDecoderService"}`).
    pub const CREATE_DECODER: u32 = 1;
    /// PR `GetSubmissionEvent`; NX2 `GetEvent`.
    pub const GET_SUBMISSION_EVENT: u32 = 3;
    /// PR and NX2 `Start`.
    pub const START: u32 = 6;
    /// PR and NX2 `Flush`.
    pub const FLUSH: u32 = 8;
    /// PR and NX2 `Configure`.
    pub const CONFIGURE: u32 = 101;
    /// PR `Submit`; NX2 `SubmitInputBuffer`.
    pub const SUBMIT: u32 = 109;
    /// PR and NX2 `Finalize`.
    pub const FINALIZE: u32 = 115;
    /// PR `InitializeByTransferMemory200`; NX2 `Initialize`.
    pub const INITIALIZE_200: u32 = 200;
    /// PR `InitializeByTransferMemory`.
    pub const INITIALIZE: u32 = 201;
    /// PR and NX2 `Stop`.
    pub const STOP: u32 = 207;
    /// PR `UnregisterImage`; NX2 `UnregisterBuffer`.
    pub const UNREGISTER_IMAGE: u32 = 212;
    /// PR `GetOutputProperties`; NX2 `GetOutputInfo`.
    pub const GET_OUTPUT_PROPERTIES: u32 = 213;
    /// PR `RegisterImage`; NX2 `RegisterBuffer`.
    pub const REGISTER_IMAGE: u32 = 311;
    /// PR `ReleaseImage`; NX2 `ReleaseBuffer`.
    pub const RELEASE_IMAGE: u32 = 314;
}

/// The most images a decoder service holds registered (PR `RegisterImage`:
/// `registered_images.size() < 8`).
pub const MAX_REGISTERED: usize = 8;
/// The most handouts waiting for a frame: starquake's own bound (PR and NX2 have none),
/// twice [`MAX_REGISTERED`], room for every image registered and handed back once more while
/// it still waits (the case PR's `ReleaseImage` comment notes: "releases images that were never
/// handed out").
pub const MAX_HANDOUTS: usize = 2 * MAX_REGISTERED;
/// The most submitted frames waiting for an image (PR `MaximumPendingInputs = 3`).
pub const MAX_PENDING: usize = 3;
/// The most finished outputs waiting for 213: starquake's own bound (PR and NX2 have none),
/// [`MAX_REGISTERED`].
pub const MAX_OUTPUTS: usize = MAX_REGISTERED;

/// 109's answer when the input queue is full (PR `Submit`: `result = Result{0x1c266f}`).
pub const VIDEO_QUEUE_FULL: ResultCode = ResultCode::from_raw(0x001C_266F);
/// 213's answer when no output is finished (PR `GetOutputProperties`: `Result{0x1c286f}`; NX2
/// `GetOutputInfo`: "the code the client's own Acquire uses for an empty ring",
/// `rb.PushRaw<u32>(0x1c286f)`).
pub const VIDEO_NO_OUTPUT: ResultCode = ResultCode::from_raw(0x001C_286F);

/// How many words 213 answers (PR `std::array<u32, 16> properties`; NX2: 16 words).
pub const OUTPUT_WORDS: usize = 16;
/// How many ns a µs is: 213's timestamp (PR `frame.timestamp * 1000`; NX2 `output->pts *
/// 1000`).
pub const NS_PER_US: u64 = 1000;
/// The mode word of a native sync handle: 0, the 32-bit wrapping comparison (PR
/// `rb.Push<u32>(0)`; NX2 `{~m_syncpoint, 0, value, 0}`).
pub const SYNC_HANDLE_MODE: u32 = 0;

/// The `SyncCounterId` of syncpoint `id`: its bitwise NOT (spec, "Ids on the wire"; PR
/// `~*operation_syncpoint`; NX2 `~m_syncpoint`). The inverse of [`crate::syncpt::syncpoint_of`].
#[must_use]
pub fn counter_id(id: SyncpointId) -> u32 {
    !u32::from(id.get())
}

/// The low 32 bits of `value`: a handout's number on the 32-bit counter.
const fn low_word(value: u64) -> u32 {
    let [a, b, c, d, ..] = value.to_le_bytes();
    u32::from_le_bytes([a, b, c, d])
}

/// What every video session shares: the handle to gsv's shared state, where the pools are and
/// through which the syncpoints are reached. It saves nothing.
pub struct VideoFamily {
    shared: Arc<Shared>,
}

impl VideoFamily {
    /// The family over `shared`.
    #[must_use]
    pub fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }
}

impl Family for VideoFamily {
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::Inconsistent(
                "the video family saves no state",
            ))
        }
    }
}

/// The handle to gsv's shared state, from the server's family.
fn shared(cx: &mut Context<'_>) -> Result<Arc<Shared>, Fault> {
    Ok(Arc::clone(&cx.family::<VideoFamily>()?.shared))
}

/// The video context: what 902, 906 and 1002 hand out. It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VideoContext;

commands! {
    impl VideoContext {}
    tipc {
        CREATE_DECODER => create_decoder,
    }
}

impl VideoContext {
    /// `CreateDecoder` (1): a new decoder service on this server, copied (the root's house rule
    /// for handles, [`crate::protocol::HANDLE_KIND`]; PR moves it, NX2 copies it).
    fn create_decoder(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.copy_session(Box::new(DecoderService::new()));
        reply.ok()
    }
}

/// What 101 named.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Configuration {
    /// The width (PR `packet[5]`).
    pub width: u32,
    /// The height (PR `packet[6]`).
    pub height: u32,
    /// The frame rate (PR `packet[7]`).
    pub frame_rate: u32,
    /// The input capacity in bytes (PR `packet[8]`).
    pub input_capacity: u32,
}

/// An output image handed out (311 or 314), waiting for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handout {
    /// The image.
    pub image: u32,
    /// Its number since the syncpoint was allocated, from 1.
    pub number: u64,
}

/// A submitted frame, waiting for a handout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    /// Its timestamp in µs.
    pub timestamp: u64,
    /// Whether it ends the stream.
    pub end: bool,
}

/// A finished frame, waiting for 213.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// The image it went into.
    pub image: u32,
    /// The configured width and height when it finished.
    pub width: u32,
    /// The configured height.
    pub height: u32,
    /// Its timestamp in µs.
    pub timestamp: u64,
    /// Whether it ends the stream.
    pub end: bool,
}

impl Output {
    /// 213's sixteen words (module docs, "The decoder service"). The frame's size is the
    /// configured size: the null decoder has no picture.
    #[must_use]
    pub fn words(&self) -> [u32; OUTPUT_WORDS] {
        let ns = self.timestamp.wrapping_mul(NS_PER_US);
        let mut words = [0; OUTPUT_WORDS];
        words[0] = self.image;
        words[1] = self.width;
        words[2] = self.height;
        words[3] = self.width;
        words[4] = self.height;
        words[6] = low_word(ns);
        words[7] = low_word(ns >> 32);
        words[8] = u32::from(self.end);
        words
    }
}

/// A decoder service with the null decoder (module docs, "The null decoder"). Plain data, saved
/// with the server; reading one back goes through `SavedDecoderService`, which refuses a state
/// the commands cannot reach ([`DecoderServiceError`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedDecoderService")]
pub struct DecoderService {
    configuration: Option<Configuration>,
    event: Option<ObjectId>,
    syncpoint: Option<u8>,
    base: u32,
    handed: u64,
    reached: u64,
    images: Vec<u32>,
    handouts: VecDeque<Handout>,
    pending: VecDeque<Input>,
    outputs: VecDeque<Output>,
}

/// The serialised form of [`DecoderService`], field for field.
#[derive(Deserialize)]
struct SavedDecoderService {
    configuration: Option<Configuration>,
    event: Option<ObjectId>,
    syncpoint: Option<u8>,
    base: u32,
    handed: u64,
    reached: u64,
    images: Vec<u32>,
    handouts: VecDeque<Handout>,
    pending: VecDeque<Input>,
    outputs: VecDeque<Output>,
}

/// Why a saved [`DecoderService`] could not be read back: it holds what no sequence of commands
/// leads to. Checked in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum DecoderServiceError {
    /// More images than [`MAX_REGISTERED`].
    #[error("a decoder service holds {found} images, more than {MAX_REGISTERED}")]
    Images {
        /// How many.
        found: usize,
    },
    /// An image id is 0 or registered twice.
    #[error("a decoder service holds image {image} twice, or image 0")]
    Image {
        /// The image.
        image: u32,
    },
    /// The syncpoint is not one below the count, or it is missing while handouts were made, or
    /// there while none were.
    #[error("a decoder service's syncpoint does not match its handouts")]
    Syncpoint,
    /// The last finished handout is past the last made.
    #[error("a decoder service finished handout {reached}, but made only {handed}")]
    Reached {
        /// The last finished.
        reached: u64,
        /// The last made.
        handed: u64,
    },
    /// More handouts, frames or outputs than their bound.
    #[error("a decoder service's queues pass their bounds")]
    Queues,
    /// A handout names an image not registered, or its number is not above the one before it
    /// (the last finished, for the first), or passes the last made.
    #[error("a decoder service's handout {index} is out of order or names an unknown image")]
    Handout {
        /// Its position in the queue.
        index: usize,
    },
    /// A frame waits while a handout and room for its output wait too: a command would have
    /// finished it.
    #[error("a decoder service has a frame, a handout and room, unfinished")]
    Unfinished,
}

impl TryFrom<SavedDecoderService> for DecoderService {
    type Error = DecoderServiceError;

    fn try_from(saved: SavedDecoderService) -> Result<Self, Self::Error> {
        let SavedDecoderService {
            configuration,
            event,
            syncpoint,
            base,
            handed,
            reached,
            images,
            handouts,
            pending,
            outputs,
        } = saved;
        if images.len() > MAX_REGISTERED {
            return Err(DecoderServiceError::Images {
                found: images.len(),
            });
        }
        for (index, &image) in images.iter().enumerate() {
            if image == 0 || images[..index].contains(&image) {
                return Err(DecoderServiceError::Image { image });
            }
        }
        let valid = syncpoint.is_none_or(|id| SyncpointId::new(id).is_some());
        if !valid || syncpoint.is_some() != (handed > 0) {
            return Err(DecoderServiceError::Syncpoint);
        }
        if reached > handed {
            return Err(DecoderServiceError::Reached { reached, handed });
        }
        if handouts.len() > MAX_HANDOUTS
            || pending.len() > MAX_PENDING
            || outputs.len() > MAX_OUTPUTS
        {
            return Err(DecoderServiceError::Queues);
        }
        let mut previous = reached;
        for (index, handout) in handouts.iter().enumerate() {
            if !images.contains(&handout.image)
                || handout.number <= previous
                || handout.number > handed
            {
                return Err(DecoderServiceError::Handout { index });
            }
            previous = handout.number;
        }
        if !pending.is_empty() && !handouts.is_empty() && outputs.len() < MAX_OUTPUTS {
            return Err(DecoderServiceError::Unfinished);
        }
        Ok(Self {
            configuration,
            event,
            syncpoint,
            base,
            handed,
            reached,
            images,
            handouts,
            pending,
            outputs,
        })
    }
}

impl Default for DecoderService {
    fn default() -> Self {
        Self::new()
    }
}

commands! {
    impl DecoderService {}
    tipc {
        GET_SUBMISSION_EVENT => get_submission_event,
        START => start,
        FLUSH => flush,
        CONFIGURE => configure [In(MapAlias)],
        SUBMIT => submit [In(MapAlias)],
        FINALIZE => finalize,
        INITIALIZE_200 => initialize,
        INITIALIZE => initialize,
        STOP => flush,
        UNREGISTER_IMAGE => unregister_image,
        GET_OUTPUT_PROPERTIES => get_output_properties,
        REGISTER_IMAGE => register_image,
        RELEASE_IMAGE => release_image,
    }
}

impl DecoderService {
    /// A fresh decoder service: nothing configured, registered or waiting, no syncpoint and no
    /// event (PR `INativeVideoCodec`'s members; NX2 `INx2VideoSession`'s).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            configuration: None,
            event: None,
            syncpoint: None,
            base: 0,
            handed: 0,
            reached: 0,
            images: Vec::new(),
            handouts: VecDeque::new(),
            pending: VecDeque::new(),
            outputs: VecDeque::new(),
        }
    }

    /// What 101 named last.
    #[must_use]
    pub const fn configuration(&self) -> Option<&Configuration> {
        self.configuration.as_ref()
    }

    /// The ready event, once 3 or 101 made it.
    #[must_use]
    pub const fn event(&self) -> Option<ObjectId> {
        self.event
    }

    /// The syncpoint that counts finished frames, once 311 allocated it.
    #[must_use]
    pub fn syncpoint(&self) -> Option<SyncpointId> {
        self.syncpoint.and_then(SyncpointId::new)
    }

    /// The registered images, in registration order.
    #[must_use]
    pub fn images(&self) -> &[u32] {
        &self.images
    }

    /// The handouts waiting for a frame, oldest first.
    pub fn handouts(&self) -> impl Iterator<Item = &Handout> {
        self.handouts.iter()
    }

    /// How many handouts were made since the syncpoint was allocated.
    #[must_use]
    pub const fn handed(&self) -> u64 {
        self.handed
    }

    /// The submitted frames waiting for a handout, oldest first.
    pub fn pending(&self) -> impl Iterator<Item = &Input> {
        self.pending.iter()
    }

    /// The finished outputs waiting for 213, oldest first.
    pub fn outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter()
    }

    /// The value the syncpoint reaches when handout `number` is finished: `base + number`,
    /// wrapping (module docs, "The null decoder").
    #[must_use]
    pub const fn value(&self, number: u64) -> u32 {
        self.base.wrapping_add(low_word(number))
    }

    /// Finishes waiting frames into waiting handouts, oldest with oldest, while the output queue
    /// has room (module docs, "The null decoder").
    fn pump(&mut self, syncpoints: &Syncpoints) {
        let Some(id) = self.syncpoint() else {
            return;
        };
        while self.outputs.len() < MAX_OUTPUTS && !self.pending.is_empty() {
            let Some(handout) = self.handouts.pop_front() else {
                break;
            };
            let Some(input) = self.pending.pop_front() else {
                break;
            };
            // Above `reached` (the queue's order), so at least 1; its low word is 0 only after
            // 2^32 handouts taken out unfinished, which leaves the counter where it is: the same
            // 32-bit value.
            syncpoints.increment_by(id, low_word(handout.number - self.reached));
            self.reached = handout.number;
            let (width, height) = self
                .configuration
                .map_or((0, 0), |config| (config.width, config.height));
            self.outputs.push_back(Output {
                image: handout.image,
                width,
                height,
                timestamp: input.timestamp,
                end: input.end,
            });
        }
    }

    /// Signals the ready event if the input queue has room, and clears it if not; nothing if
    /// no command has made it.
    fn update_ready(&self, cx: &mut Context<'_>) -> Result<(), Fault> {
        let Some(event) = self.event else {
            return Ok(());
        };
        if self.pending.len() < MAX_PENDING {
            cx.signal_event(event)
        } else {
            cx.clear_event(event)
        }
    }

    /// The ready event, made the first time.
    fn ready_event(&mut self, cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
        if let Some(event) = self.event {
            return Ok(event);
        }
        let event = cx.create_event()?;
        self.event = Some(event);
        Ok(event)
    }

    /// Hands image `image` out with the next number, after the checks its command made, and
    /// finishes what it can. Allocates the syncpoint the first time: the last step that can
    /// fail. Returns the handout's value.
    fn hand_out(
        &mut self,
        cx: &mut Context<'_>,
        shared: &Shared,
        image: u32,
    ) -> Result<(SyncpointId, u32), Fault> {
        if self.handouts.len() >= MAX_HANDOUTS {
            return Err(KernelError::OutOfResource.into());
        }
        let syncpoints = shared.syncpoints();
        let id = if let Some(id) = self.syncpoint() {
            id
        } else {
            let id = syncpoints.allocate().map_err(|error| {
                tracing::warn!(%error, "gsv:video: no syncpoint for a decoder service");
                Fault::from(KernelError::OutOfResource)
            })?;
            self.syncpoint = Some(id.get());
            self.base = syncpoints.read(id);
            self.handed = 0;
            self.reached = 0;
            id
        };
        self.handed += 1;
        let number = self.handed;
        if !self.images.contains(&image) {
            self.images.push(image);
        }
        self.handouts.push_back(Handout { image, number });
        self.pump(syncpoints);
        self.update_ready(cx)?;
        Ok((id, self.value(number)))
    }

    /// Writes a native sync handle and two `u64` zeros (PR's eight words).
    fn write_handle(reply: &mut Reply, id: SyncpointId, value: u32) -> Outcome {
        reply
            .payload()
            .write_u32(counter_id(id))
            .write_u32(SYNC_HANDLE_MODE)
            .write_u64(u64::from(value))
            .write_u64(0)
            .write_u64(0);
        reply.ok()
    }

    /// `GetSubmissionEvent` (3): the ready event, copied.
    fn get_submission_event(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let event = this.ready_event(cx)?;
        this.update_ready(cx)?;
        reply.copy_handle(event);
        reply.ok()
    }

    /// `Start` (6).
    fn start(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        this.update_ready(cx)?;
        reply.ok()
    }

    /// `Flush` (8) and `Stop` (207): drops the frames waiting for an image.
    fn flush(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        this.pending.clear();
        this.update_ready(cx)?;
        reply.ok()
    }

    /// `Configure` (101). Words: width 5, height 6, frame rate 7, input capacity 8. out: `u64`
    /// 0.
    fn configure(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let configuration = Configuration {
            width: words.read_u32()?,
            height: words.read_u32()?,
            frame_rate: words.read_u32()?,
            input_capacity: words.read_u32()?,
        };
        let name = req.in_buffer(cx, 0)?.len();
        tracing::info!(
            width = configuration.width,
            height = configuration.height,
            frame_rate = configuration.frame_rate,
            name_bytes = name,
            "gsv:video: a decoder configured; frames finish with no picture (decision 10)"
        );
        this.ready_event(cx)?;
        this.configuration = Some(configuration);
        this.update_ready(cx)?;
        reply.payload().write_u64(0);
        reply.ok()
    }

    /// `Submit` (109). Words: timestamp 5–6, end of stream 7 (low byte). out: `u64` the frame's
    /// bytes.
    fn submit(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut words = req.payload();
        let timestamp = words.read_u64()?;
        let end = low_byte(words.read_u32()?) != 0;
        let bytes = req.in_buffer(cx, 0)?.len();
        if this.pending.len() >= MAX_PENDING {
            return Err(VIDEO_QUEUE_FULL.into());
        }
        this.pending.push_back(Input { timestamp, end });
        let shared = shared(cx)?;
        this.pump(shared.syncpoints());
        this.update_ready(cx)?;
        reply.payload().write_u64(bytes);
        reply.ok()
    }

    /// `Finalize` (115): forgets everything but the event, which is cleared, and frees the
    /// syncpoint.
    fn finalize(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if let Some(id) = this.syncpoint() {
            let shared = shared(cx)?;
            if let Err(error) = shared.syncpoints().free(id) {
                tracing::error!(%error, "gsv:video: a decoder service's syncpoint was not allocated");
            }
        }
        let event = this.event;
        *this = Self::new();
        this.event = event;
        if let Some(event) = event {
            cx.clear_event(event)?;
        }
        reply.ok()
    }

    /// `InitializeByTransferMemory200` (200) and `InitializeByTransferMemory` (201). Copy handle
    /// 0; words: size 4–5, codec 6, input format 7, output format 8, and 201's output mode 9.
    /// Read and logged; nothing is checked or kept (module docs, "Where this differs").
    fn initialize(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        req.copy_handle(0)?;
        let mut words = req.payload();
        let size = words.read_u64()?;
        let codec = words.read_u32()?;
        let input = words.read_u32()?;
        let output = words.read_u32()?;
        if req.command_id() == INITIALIZE {
            words.read_u32()?;
        }
        tracing::info!(
            size,
            codec,
            input,
            output,
            "gsv:video: a decoder initialised; nothing is decoded (decision 10)"
        );
        reply.ok()
    }

    /// `UnregisterImage` (212). Words: image 2.
    fn unregister_image(
        this: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let image = req.payload().read_u32()?;
        let at = this
            .images
            .iter()
            .position(|&registered| registered == image)
            .ok_or(Fault::from(KernelError::NotFound))?;
        this.images.remove(at);
        this.handouts.retain(|handout| handout.image != image);
        reply.ok()
    }

    /// `GetOutputProperties` (213): the oldest finished output's sixteen words, or
    /// [`VIDEO_NO_OUTPUT`].
    fn get_output_properties(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let output = this
            .outputs
            .pop_front()
            .ok_or(Fault::from(VIDEO_NO_OUTPUT))?;
        let shared = shared(cx)?;
        this.pump(shared.syncpoints());
        this.update_ready(cx)?;
        let out = reply.payload();
        for word in output.words() {
            out.write_u32(word);
        }
        reply.ok()
    }

    /// `RegisterImage` (311). Words: image 2. out: a native sync handle. Checks, in PR's order:
    /// [`MAX_REGISTERED`] images, `OutOfResource`; registered already, `InvalidState`; no pool
    /// image with the id, `NotFound`; [`MAX_HANDOUTS`] handouts, or no syncpoint left,
    /// `OutOfResource`.
    fn register_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let image = req.payload().read_u32()?;
        if this.images.len() >= MAX_REGISTERED {
            return Err(KernelError::OutOfResource.into());
        }
        if this.images.contains(&image) {
            return Err(KernelError::InvalidState.into());
        }
        let shared = shared(cx)?;
        if shared.lock().native_objects().pool_image(image).is_none() {
            return Err(KernelError::NotFound.into());
        }
        let (id, value) = this.hand_out(cx, &shared, image)?;
        Self::write_handle(reply, id, value)
    }

    /// `ReleaseImage` (314). Words: image 2. out: a native sync handle. Checks: not registered,
    /// `NotFound`; [`MAX_HANDOUTS`] handouts, `OutOfResource`.
    fn release_image(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let image = req.payload().read_u32()?;
        if !this.images.contains(&image) {
            return Err(KernelError::NotFound.into());
        }
        let shared = shared(cx)?;
        let (id, value) = this.hand_out(cx, &shared, image)?;
        Self::write_handle(reply, id, value)
    }
}

/// The interfaces a video server's sessions can be.
#[must_use]
pub fn video_interfaces() -> Interfaces {
    Interfaces::new()
        .with::<VideoContext>()
        .with::<DecoderService>()
}

/// The video family's server over gsv's `shared` state (module docs, "The server"): every new
/// session a [`VideoContext`].
#[must_use]
pub fn video_server(shared: Arc<Shared>, policy: UnknownPolicy) -> Server {
    Server::new(VIDEO_NAME, policy)
        .with_root(|| Box::new(VideoContext))
        .with_interfaces(video_interfaces())
        .with_family(VideoFamily::new(shared))
}
