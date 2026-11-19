//! `/dev/nvhost-gpu`: GPU channels, their object classes and events, and GPFIFO submission, with
//! a deferred reply when the GPU's queue for the channel is full
//! (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv" and decision 12; port
//! slice 11).
//!
//! * [`submits`]: the submissions that wait for room in gsv's shared state, and their snapshot.
//!
//! Ported from Eden (GPL-3.0-or-later) @ cf654052,
//! `eden/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` and `nvhost_gpu.h` (`Ioctl1`,
//! `Ioctl2`, `AllocGPFIFOEx2`, `AllocateObjectContext`, `GetObjectContextClassNumberIndex`,
//! `SubmitGPFIFOImpl`, `SubmitGPFIFOBase1`, `SubmitGPFIFOBase2`, `QueryEvent`, the argument
//! structs) and `nvdata.h` (`NvFence`), with Eden-NX2's Switch 2 changes (GPL-3.0-or-later,
//! Copyright 2026 Eden Emulator Project) @ 50fa86bb,
//! `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` (the Ampere and class-0 mapping
//! in `AllocateObjectContext`, `BuildIncrementCountCommandList`, `QueryEvent`'s extra events),
//! and Eden PR #4588's readings (GPL-3.0-or-later) @ e5fb3b97,
//! `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` (`QueryEvent`'s event 7,
//! `GetNativeGpuContextSlot` in `native_gpu_context.h`; `ZCullBind` and `IoctlZCullBind`, the
//! same in Eden and NX2). "Row N" is a row of
//! `eden-nx2/NX2_NOTES.md` @ 50fa86bb: rows 26 (channel setup: Ampere and class-0 objects), 31
//! (the Ampere classes C797, C7C0, C7B5 and class 0 bound), 32 (NVN2 sets bit 8 and expects the
//! increments though its commands hold none) and 34 (a per-channel event, `QueryEvent` 7), all
//! observed. libnx (`libnx/nx/source/nvidia/ioctl/nvchannel.c` @ b156b477) sends the same
//! numbers: `_NV_IOWR(0x48, 0x08, ..)` with the entries after the struct, `nvIoctl2` with
//! `_NV_IOWR(0x48, 0x1B, ..)` and the entries inline, `_NV_IOWR(0x48, 0x09, ..)`,
//! `_NV_IOWR(0x48, 0x1A, ..)`, and `_NV_IOWR(0x48, 0x0B, ..)` (`nvioctlChannel_ZCullBind`, with
//! the same 16-byte struct).
//!
//! # The channel
//!
//! Each nvhost-gpu fd is one channel, as each Eden `nvhost_gpu` device is (one per `Open`). Its
//! state ([`Channel`]) is made by its first ioctl or event query and goes when its fd closes:
//!
//! - its **syncpoint**, allocated by `AllocGPFIFOEx2` from gsv's `Syncpoints` (the spec:
//!   "Channel creation (`AllocGPFIFOEx2`) allocates a syncpoint"; Eden allocates it when the fd
//!   opens, a listed divergence) and freed when the fd closes;
//! - its **address space**, which nvhost-as-gpu's `BindChannel` binds it to
//!   ([`super::as_gpu`]: the binding holds the space);
//! - its **GPU channel** ([`ChannelFront::create_channel`]), made once it has both, at
//!   `AllocGPFIFOEx2` if the channel is bound by then, otherwise at its first submission, as
//!   Eden's `SubmitGPFIFOImpl` initialises a channel lazily ("Lazily initialize channel when
//!   address space is available"); destroyed when the fd closes;
//! - its **object classes** (`AllocObjCtx`), one per engine slot, each given to the GPU channel
//!   as it is bound, or in the order bound when the GPU channel is made;
//! - its **events** for `QueryEvent`: the ids go when the fd closes, but the kernel events they
//!   name are not freed (see "Where this differs").
//!
//! # The ioctls
//!
//! Group `'H'` ([`GROUP`]); each argument a fixed struct ([`super::ioctl::Fixed`]), given back
//! whole whatever the result, every field the ioctl does not set as the caller sent it
//! (`WrapFixed`, `WrapFixedVariable`, `WrapFixedInlIn`).
//!
//! | Variant | Cmd | Ioctl | Argument (offset: field) |
//! |---|---|---|---|
//! | `Ioctl` | 0x1A | `AllocGPFIFOEx2` | 0: entries, 4: jobs, 8: flags, 12: fence out (`s32` id, 16: `u32` value), 20: three reserved words |
//! | `Ioctl` | 0x9 | `AllocObjCtx` | 0: class, 4: flags, 8: object id (`u64`) |
//! | `Ioctl` | 0x8 | `SubmitGPFIFO` | 0: address (`u64`, not read), 8: entry count, 12: flags, 16: fence (`s32` id, 20: `u32` value); the entries after the struct, 8 bytes each |
//! | `Ioctl2` | 0x1B | `SubmitGPFIFO` (libnx `KickoffPb`) | as 0x8, the entries in the inline input (in buffer 1) |
//! | `Ioctl` | 0x3 | `ChannelSetTimeout` | 0: timeout |
//! | `Ioctl` | 0xC | `SetErrorNotifier` | 0: offset (`u64`), 8: size (`u64`), 16: mem, 20: padding |
//! | `Ioctl` | 0x1F, only [`SET_PREEMPTION_MODES`] | `SetPreemptionModes` (PR's) | 0: graphics, 4: compute |
//! | `Ioctl` | 0xB | `ZCullBind` | 0: GPU address (`u64`), 8: mode, 12: padding |
//! | `Ioctl` | group `'G'` ([`CLIENT_GROUP`]) 0x14 | `SetClientData` | 0: data (`u64`) |
//! | `Ioctl` | group `'G'` 0x15 | `GetClientData` | 0: data (`u64`) |
//!
//! Every other number, and `Ioctl3`, goes to the server's unknown-ioctl policy (see "Not built").
//! The title of issue #300 sends `0x40044803`, `0xc018480c`, `0xc008481f` and `0x40084714`; no
//! title is known to send `GetClientData`, which is built because it reads what
//! `SetClientData` keeps (issue #300's scope). Its number is matched on the group and the
//! command alone, as PR's `Ioctl1` does. The title of issue #306 sends `0xc010480b`
//! (`ZCullBind`), matched the same way.
//!
//! **`AllocGPFIFOEx2`**, the first failing check answering: the channel has a syncpoint already,
//! `AlreadyAllocated` (Eden: `channel_state->initialized`); no syncpoint is free,
//! `InsufficientMemory` (a guess: Eden's allocator asserts); the channel is bound and the GPU
//! refuses a channel, `InsufficientMemory` (a guess), and the syncpoint is freed again. Then the
//! fence out is the syncpoint and its maximum (`GetSyncpointFence`). Nothing else is read.
//!
//! **`AllocObjCtx`** (NX2's, row 26): non-zero flags are logged and given back as 0 (Eden:
//! `params.flags = allowed_mask`). The class is mapped onto Eden's engine slots ([`slot_class`]:
//! the Ampere, Turing and Volta classes onto the Maxwell ones; class 0 onto the first free of
//! 3D, compute, copy, 2D and inline-to-memory), and the mapped class is what comes back. Then: a
//! class with no slot, `BadParameter`; a slot taken, `AlreadyAllocated`; the GPU refusing the
//! class on a channel already made, `BadParameter` (a guess: Eden cannot refuse), and nothing
//! recorded. The GPU is given the class as the caller sent it (the spec: "binding an object class
//! goes to the front unchanged"), class 0 as the class it was mapped onto.
//!
//! **`SubmitGPFIFO`**, the steps in order (Eden `SubmitGPFIFOBase1`, `SubmitGPFIFOBase2`,
//! `SubmitGPFIFOImpl`; the spec's steps 1 to 4); the first failing check answers, and nothing
//! changes:
//!
//! 1. the count above the entries the request carries (whole 8-byte entries after the struct for
//!    0x8, in the inline input for 0x1B): `InvalidSize` (Eden: `params.num_entries >
//!    commands.size()`);
//! 2. the count above [`CHANNEL_QUEUE_ENTRIES`]: `InvalidSize` (starquake's: such a submission
//!    could never be queued);
//! 3. flags bit 0 and bit 8 together: `BadParameter` (Eden: `fence_wait` with
//!    `increment_value`);
//! 4. the increment overflows 32 bits: `BadParameter` (starquake's, below);
//! 5. bit 0 with a wait fence whose id is not a syncpoint: `BadParameter` (starquake's: Eden
//!    indexes its table with it);
//! 6. the channel has no syncpoint, or no GPU channel and no address space to make one in:
//!    `InvalidState` (starquake's: Eden submits to a channel it never initialised); the GPU
//!    refusing the channel made now: `InsufficientMemory` (a guess);
//! 7. the submission would take the channel's waiting submissions past
//!    [`submits::MAX_WAITING_ENTRIES`] ([`submits::Submits::admits`]): `Busy` (the owner's ruling
//!    of 2026-11-17, decision log: "Cap a channel's waiting submissions at 65,536 entries; past
//!    it, answer Busy"; [`submits`], "The cap"). Only a channel whose submissions wait can be
//!    refused; with nothing waiting, check 2 keeps every submission within the cap.
//!
//! Then the fence ([`plan`]): the increment is 2 if bit 1 ([`FENCE_INCREMENT`]) plus the fence
//! value if bit 8 ([`INCREMENT_VALUE`]); the channel syncpoint's maximum is raised by it, and the
//! fence given back is (the channel's syncpoint, the new maximum), the flags word 0 (Eden:
//! `flags.raw = 0`). The GPU front is handed one [`Submission`]: the entries as sent, the wait
//! fence if bit 0 ([`FENCE_WAIT`]), the increment as the number of increments it appends after
//! them (row 32), and the wait-for-idle flag, set unless bit 4 ([`SUPPRESS_WFI`]). The reply does
//! not wait for the GPU.
//!
//! **A full queue** (decision 12). If the front answers [`Submitted::Full`], or submissions of
//! the channel wait already (so this one cannot pass them), the request takes a reply token and
//! its submission waits in gsv's shared state ([`submits::Submits`]); the argument with its fence
//! is written to the caller's out buffer at once, and only the result word waits. When the
//! front reports room (`GpuCompletion::GpuQueueRoom`), the timer thread's drain gives the
//! channel's waiting submissions to the front again, oldest first, and completes the reply of
//! each one queued, until the front answers `Full` again. Entries reach the front in the order
//! the submissions came, whether they waited or not. A channel's waiting submissions are capped
//! (check 7 above): past the cap the request answers `Busy` and gets no fence.
//!
//! The syncpoint's maximum is raised before the reply is written, so a submission that must wait
//! waits even if its out buffer cannot be written or no reply token can be taken: the request
//! answers that fault, and the submission waits with no token, its fence still reached once it
//! is queued (`a_waiting_submission_with_an_unwritable_out_buffer_still_counts`;
//! [`submits`], "A submission without a token").
//!
//! # The settings
//!
//! Six ioctls set or read a channel's settings (PR `nvhost_gpu.cpp` @ e5fb3b97,
//! `ChannelSetTimeout`, `SetErrorNotifier`, `SetPreemptionModes`, `ZCullBind`, `SetClientData`,
//! `GetClientData`, each through `WrapFixed`). Each gives its struct back whole, the fields it
//! does not set as sent, whatever the result. None calls the GPU front.
//!
//! - **`ChannelSetTimeout`**: logged at info, nothing kept, `Success` (PR, NX2 and Eden alike).
//! - **`SetErrorNotifier`**: the offset, size and nvmap handle are kept in the channel
//!   ([`Channel::error_notifier`]), `Success`. Nothing reads them yet: an error notifier is
//!   written when a channel fails, which is #12's (`GpuFront::channel_error`). PR, NX2 and Eden
//!   log the call and keep nothing ("(STUBBED)"); keeping it changes nothing the guest sees.
//! - **`SetPreemptionModes`** (PR's, `UsesNativeAbi() && command.raw == 0xc008481f`; nvsrv is
//!   reached only through gsv 1402, by Switch 2 processes, so every caller is on the native ABI,
//!   as for tsg-gpu, `super`, "Devices and fds"): the checks in order, the first failing
//!   answering: the channel's GPU channel is not made, `InvalidState` (PR:
//!   `!channel_state->initialized`); the graphics mode above [`preemption::GRAPHICS_MAX`] or the
//!   compute mode above [`preemption::COMPUTE_MAX`], `NotSupported`. Then each mode that is not
//!   [`preemption::KEEP`] is kept ([`Channel::preemption`], each [`preemption::DEFAULT`] until
//!   set), `Success`. Nothing reads them yet (PR: "Context switches follow complete commands").
//! - **`ZCullBind`**: the GPU address and the mode are kept ([`Channel::zcull`]), `Success`,
//!   whatever their values: PR, NX2 and Eden keep the struct (`zcull_params = params`) and check
//!   nothing, and PR's comment names modes 0 to 3 (`0=global, 1=no_ctxsw,
//!   2=separate_buffer, 3=part_of_regular_buf`) without refusing others. The struct is read as far as the input
//!   goes and the rest is zero, as for every `WrapFixed` ioctl, so no input size is refused.
//!   Nothing reads the bind yet, in starquake or in the references (no reference reads
//!   `zcull_params` after storing it): the address is where the channel's zcull context buffer
//!   lives (its size is ctrl-gpu's `ZCullGetCtxSize`, [`super::ctrl_gpu`]), engine state that the
//!   GPU front (#12) would read when it saves and restores a channel's zcull context.
//! - **`SetClientData`**: the `u64` is kept ([`Channel::client_data`]), `Success`.
//!   **`GetClientData`**: the struct's data set to the kept one, 0 if none was set (PR
//!   `user_data{}`), `Success`. That is the only read of anything the settings keep.
//!
//! `ChannelSetTimeout` makes no channel; the setters that keep something make the fd's channel
//! if it has none yet, as an event query does; `GetClientData` makes none.
//!
//! The logs (`ChannelSetTimeout`'s at info, `SetErrorNotifier`'s and `ZCullBind`'s at debug, a
//! refused `SetPreemptionModes` at warn) are log-only: no test captures them, so a change to their level
//! or text is not observed.
//!
//! # Events
//!
//! `QueryEvent` on an nvhost-gpu fd answers ids 1 (SM exception breakpoint interrupt), 2 (SM
//! exception breakpoint pause) and 3 (error notifier), Eden's, and 7, PR's, which is the error
//! notifier's event too (PR: "The native SDK requests this event immediately before configuring
//! `CHANNEL_SET_ERROR_NOTIFIER`"; row 34: NVN asks for 7). Each is made the first time it is asked
//! for and copied out with `Success`; any other id answers `BadParameter`, Eden's answer for an
//! event the device does not have. Nothing signals them yet: the GPU front reports a stopped
//! channel's error (`GpuFront::channel_error`, `starquake_gpu::interface`), and nvsrv does not
//! ask it yet.
//!
//! # Conflicts
//!
//! Each is picked by the house rule of decision 8 ("where one answer contains the other, the
//! larger"), except the event ids and `SetPreemptionModes`, and pinned by the test named, in
//! `gpu/tests.rs` unless another file is given. The spec's verification item 1 lists the event
//! ids; it does not list the others yet. The open question with each settles it.
//! `ZCullBind` has none: Eden's, NX2's and PR's handlers and structs are the same.
//!
//! - **The class `AllocObjCtx` gives back.** NX2 gives back the Maxwell class it mapped onto
//!   (`params.class_num = mapped`); PR gives back the class as sent. NX2's is used, the one a
//!   title ran on (row 26). Open question: does NvRm read the class back?
//!   (`alloc_obj_ctx_gives_back_the_mapped_class`.)
//! - **Which classes `AllocObjCtx` takes.** PR takes C797, C7C0, C7B5 and C56F; NX2 also the
//!   Turing and Volta classes, C76F and class 0. NX2's, the larger
//!   (`every_mapped_class_is_taken_and_others_refused`).
//! - **Event ids.** NX2 makes an event for every id it is asked (`QueryEvent`'s `default:`
//!   with `g_nx2_driver`); PR answers 1, 2, 3 and 7. NX2's answer contains PR's, so decision 8's
//!   rule picks NX2's; this **departs from decision 8** and takes PR's, because NX2's makes a
//!   new kernel event for each new id a title sends, without bound. The owner accepted the
//!   departure on 2026-11-17 (decision log: "nvhost-gpu answers event ids 1, 2, 3 and 7 only: an
//!   accepted departure from decision 8"). A listed divergence too. Open question, for a trace
//!   (the spec's verification item 1): does NvRm ask for any id besides 7?
//!   (`events_1_2_3_and_7_are_answered`.)
//! - **`SetPreemptionModes` (0x1F).** PR has a handler (above); NX2 has none, and the number
//!   reaches its catch-all for unknown Switch 2 ioctls (`nx2_stub_device.h`, `Nx2UnknownIoctl`:
//!   `Success` and the input echoed; row 26: "logged echo for unknown ioctls", observed). The
//!   spec replaces that catch-all with the services framework's unknown-ioctl policy
//!   ("Channels and submission: nvsrv": "NX2 echoed the input instead, which the framework
//!   forbids after row 96"), so the one handler, PR's, is ported, with its refusals. **Owner
//!   question:** read as decision 8's "where one answer contains the other, the larger", NX2's
//!   `Success` for every argument contains PR's; the code takes PR's because NX2's is not a
//!   reading of this ioctl. Open question, for a trace: does NvRm send it before its channel is
//!   made (PR answers `InvalidState`) or with modes above PR's limits (`NotSupported`)?
//!   (`gpu/tests/settings.rs`: `preemption_modes_check_the_channel_then_the_modes`.)
//! - **Increments past 64.** PR refuses a submission (`BadParameter`) when its ABI is the native
//!   one, flag bit 8 is set and bit 1 is clear, and the fence value is above 64
//!   (`SubmitGPFIFOImpl`: `native_completion = UsesNativeAbi() && increment_value &&
//!   !fence_increment ? value : 0; if (native_completion > 64)`); with bit 1 set, or bit 8
//!   clear, it has no limit. NX2 takes any value. NX2's, the larger
//!   (`a_large_increment_value_is_taken`).
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `gpu/tests.rs`:
//!
//! | Case | Eden (and NX2, PR) | Here |
//! |---|---|---|
//! | When the syncpoint is allocated | when the fd opens | at `AllocGPFIFOEx2`, as the spec says (`the_syncpoint_comes_with_alloc_gpfifo_ex2`) |
//! | A second `AllocGPFIFOEx2` before the channel is bound | succeeds again | `AlreadyAllocated` (`a_second_alloc_gpfifo_ex2_is_already_allocated`) |
//! | Bits 1 and 8 together | the maximum raised by 2 plus the value, but only 2 increments pushed (NX2 appends its count only without bit 1) | the front is asked for all of them, so the fence is reached (`fence_arithmetic_for_every_flag_combination`) |
//! | An increment past `u32::MAX` | wraps | `BadParameter` (`an_overflowing_increment_is_refused`) |
//! | A wait fence on a reached syncpoint | no wait pushed | the wait goes to the front, which checks it (`fence_arithmetic_for_every_flag_combination`) |
//! | A wait fence id that is not a syncpoint | out-of-range table access | `BadParameter` (`a_wait_on_no_syncpoint_is_refused`) |
//! | A submission before `AllocGPFIFOEx2`, or unbound | pushed to an uninitialised channel | `InvalidState` (`a_submission_needs_a_syncpoint_and_a_space`) |
//! | More entries than [`CHANNEL_QUEUE_ENTRIES`] in one submission | taken | `InvalidSize` (`a_submission_larger_than_a_queue_is_refused`) |
//! | A full queue | no queue bound | the reply waits for room (`a_full_queue_defers_and_room_completes`) |
//! | A submission past the cap on a channel's waiting submissions | no cap: nothing waits | `Busy`, before the fence is raised; a submission with no entries counts as one ([`submits`], "The cap"; `the_cap_takes_a_submission_that_reaches_it_and_refuses_one_past`) |
//! | The out buffer of a submission that waits | written when the ioctl returns | written at once, before the reply (`a_waiting_submission_writes_its_fence_at_once`) |
//! | Closing a channel whose submissions wait | not possible: Eden never waits | each waiting reply answers `NotImplemented`, as a request on a closed fd does; the out buffer keeps the fence (`closing_a_channel_answers_its_waiting_submissions`) |
//! | `QueryEvent` with any other id | NX2: a new event | `BadParameter` (`events_1_2_3_and_7_are_answered`) |
//! | The channel's kernel events when its fd closes | freed (`~nvhost_gpu`: `events_interface.FreeEvent` of each) | not freed: each open, `QueryEvent` 1, 2 and 3, close leaves three kernel events the server made, with no bound over the run (`a_closed_channel_s_events_are_not_freed`) |
//! | `SubmitGPFIFO` 0x1B on `Ioctl` (entries read from the guest's memory at the struct's address) | served | not served: the unknown-ioctl policy (`submissions_from_memory_go_to_the_policy`) |
//! | `SetErrorNotifier`'s arguments | logged, not kept | kept in the channel and saved; nothing reads them, so the guest sees no difference (`gpu/tests/settings.rs`: `the_error_notifier_is_kept`) |
//!
//! # What is recorded and what is saved
//!
//! Every request is one IPC event, recorded by the service framework with its payload (the
//! entries included: spec, "Submitted GPFIFO entries are part of the IPC payload"); a waiting
//! reply is recorded when it is delivered. The front's `GpuQueueRoom` comes back as a
//! `QueueRoom` note the drain records before it acts on it.
//!
//! What is **not** recorded: the front's [`Submitted::Full`]. Whether a submission waits is
//! decided at ioctl time by the front's answer, which depends on how far the GPU got, host
//! timing that no note records (the spec has a `QueueRoom` note, none for `Full`). So a replay
//! that drives a real front can defer a different set of submissions than the run did: the
//! recorded requests and notes alone do not fix what waits. Closing this gap needs a note for `Full` or a replay front that answers from
//! the record: a question for the spec.
//!
//! The channels ([`NvhostGpu`]) are plain data saved with the nvsrv state: per fd the syncpoint,
//! the GPU channel and the space it was made in, the classes, the events and the settings (the
//! client data, the error notifier, the preemption modes, the zcull bind), checked on restore ([`GpuStateError`]:
//! the preemption modes against what `SetPreemptionModes` can set; any client data, any error
//! notifier and any zcull bind can be set, so none of them is checked; and against the open files and the
//! bindings by [`super::NvsrvStateError`]). The waiting submissions are saved with gsv's shared
//! state by their own snapshot ([`submits`]), as ids (spec: "outstanding tickets and deferred
//! replies, as ids").
//!
//! Not checked here, but by [`NvhostGpu::check_syncpoints`] when a restore composes nvsrv's state
//! with the restored `Syncpoints`: that each channel's syncpoint is allocated. Not checked at all
//! yet: that the GPU channels exist on the restored front (the front's engine state is #12's;
//! the wiring slice, 16, makes them again), that an event is one the kernel holds, and that a
//! waiting submission's channel is a channel of the restored nvsrv state.
//!
//! # Not built
//!
//! The rest of Eden's nvhost-gpu: `SetNVMAPfd` (0x1), `SetChannelPriority`
//! (0xD), `AllocGPFIFOEx` (0x18), `ChannelSetTimeslice` (0x1D), `GetWaitbase` (group 0, 0x3),
//! 0x1F with any number but [`SET_PREEMPTION_MODES`], 0x20 (row 26 names 0x1F and 0x20), and
//! 0x1B on `Ioctl`. They go to the unknown-ioctl policy until a title needs one. What binding a
//! channel to a timeslice group does to the channel (PR hands it the group's macro context) is
//! engine state, #12's.

pub mod submits;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::classes::{cl902d, cla140, clc7b5, clc7c0, clc76f, clc797};
use starquake_gpu::fifo::GP_ENTRY_BYTES;
use starquake_gpu::interface::{ChannelId, Fence, SyncpointId};
use starquake_gpu::syncpoint::{FIRST_ALLOCATABLE, RESERVED_SYNCPOINTS, Syncpoints};
use starquake_kernel::id::ObjectId;

use super::as_gpu::AsGpu;
use super::ioctl::{Fixed, Ioctl};
use super::{IoctlAnswer, IoctlCall, IoctlVariant, NvResult};
use crate::gpu::AsId;
use submits::Submits;

/// nvhost-gpu's ioctl group, `'H'` (`nvhost_gpu::Ioctl1`: `case 'H':`).
pub const GROUP: u32 = 0x48;

/// nvhost-gpu's commands in the group (`nvhost_gpu::Ioctl1` and `Ioctl2`).
pub mod command {
    /// `SubmitGPFIFO` with the entries after the struct (`case 0x8:`, `SubmitGPFIFOBase1`).
    pub const SUBMIT_GPFIFO: u32 = 0x8;
    /// `AllocObjCtx` (`case 0x9:`, `AllocateObjectContext`).
    pub const ALLOC_OBJ_CTX: u32 = 0x9;
    /// `AllocGPFIFOEx2` (`case 0x1a:`).
    pub const ALLOC_GPFIFO_EX2: u32 = 0x1A;
    /// `SubmitGPFIFO` with the entries inline, on `Ioctl2` (`case 0x1b:`, `SubmitGPFIFOBase2`;
    /// libnx `nvioctlChannel_KickoffPb`).
    pub const SUBMIT_GPFIFO_INLINE: u32 = 0x1B;
    /// `ChannelSetTimeout` (`case 0x3:`; libnx `nvioctlChannel_SetTimeout`, `_NV_IOW(0x48,
    /// 0x03, ..)`).
    pub const CHANNEL_SET_TIMEOUT: u32 = 0x3;
    /// `SetErrorNotifier` (`case 0xc:`; libnx `nvioctlChannel_SetErrorNotifier`,
    /// `_NV_IOWR(0x48, 0x0C, ..)`).
    pub const SET_ERROR_NOTIFIER: u32 = 0xC;
    /// PR's `SetPreemptionModes` (`case 0x1f:`, answered only for [`super::SET_PREEMPTION_MODES`]
    /// on the native ABI).
    pub const SET_PREEMPTION_MODES: u32 = 0x1F;
    /// `ZCullBind` (`case 0xb:`, through `WrapFixed`; the title of issue #306 sends
    /// `0xc010480b`).
    pub const ZCULL_BIND: u32 = 0xB;
}

/// nvhost-gpu's second group, `'G'`, which holds the client data pair (`nvhost_gpu::Ioctl1`:
/// `case 'G':`).
pub const CLIENT_GROUP: u32 = 0x47;

/// nvhost-gpu's commands in [`CLIENT_GROUP`].
pub mod client_command {
    /// `SetClientData` (`case 0x14:`; libnx `nvioctlChannel_SetUserData`, `_NV_IOW(0x47, 0x14,
    /// ..)`).
    pub const SET_CLIENT_DATA: u32 = 0x14;
    /// `GetClientData` (`case 0x15:`).
    pub const GET_CLIENT_DATA: u32 = 0x15;
}

/// The one `SetPreemptionModes` number PR answers: `UsesNativeAbi() && command.raw ==
/// 0xc008481f` (both directions, 8 bytes); any other number with command 0x1F is PR's
/// `UNIMPLEMENTED_MSG`, here the unknown-ioctl policy.
pub const SET_PREEMPTION_MODES: u32 = 0xC008_481F;

/// The preemption modes `SetPreemptionModes` takes (PR `nvhost_gpu.cpp`, `SetPreemptionModes`;
/// `nvhost_gpu.h`, `graphics_preemption_mode{1}`, `compute_preemption_mode{1}`).
pub mod preemption {
    /// Both modes a channel starts with: `{1}`.
    pub const DEFAULT: u32 = 1;
    /// "Zero retains the current mode."
    pub const KEEP: u32 = 0;
    /// The largest graphics mode: `params.graphics > 1` is `NotSupported`.
    pub const GRAPHICS_MAX: u32 = 1;
    /// The largest compute mode: `params.compute > 2` is `NotSupported`.
    pub const COMPUTE_MAX: u32 = 2;
}

/// Where each field of each argument struct is, and each struct's size (`nvhost_gpu.h`:
/// `IoctlAllocGpfifoEx`, `IoctlAllocObjCtx`, `IoctlSubmitGpfifo` and their `static_assert`ed
/// sizes; `nvdata.h`: `NvFence { s32 id; u32 value; }`; little-endian `u32`s unless noted).
pub mod layout {
    /// `IoctlAllocGpfifoEx`: 32 bytes.
    pub const ALLOC_GPFIFO_BYTES: usize = 32;
    /// `num_entries`.
    pub const ALLOC_NUM_ENTRIES: usize = 0;
    /// `num_jobs`.
    pub const ALLOC_NUM_JOBS: usize = 4;
    /// `flags`.
    pub const ALLOC_FLAGS: usize = 8;
    /// `fence_out.id`, an `s32`.
    pub const ALLOC_FENCE_ID: usize = 12;
    /// `fence_out.value`.
    pub const ALLOC_FENCE_VALUE: usize = 16;
    /// `reserved`, three words.
    pub const ALLOC_RESERVED: usize = 20;

    /// `IoctlAllocObjCtx`: 16 bytes.
    pub const OBJ_CTX_BYTES: usize = 16;
    /// `class_num`.
    pub const OBJ_CLASS: usize = 0;
    /// `flags`.
    pub const OBJ_FLAGS: usize = 4;
    /// `obj_id`, a `u64` nothing reads.
    pub const OBJ_ID: usize = 8;

    /// `IoctlSubmitGpfifo`: 24 bytes (`16 + sizeof(NvFence)`).
    pub const SUBMIT_BYTES: usize = 24;
    /// `address`, a `u64` nothing reads here.
    pub const SUBMIT_ADDRESS: usize = 0;
    /// `num_entries`.
    pub const SUBMIT_NUM_ENTRIES: usize = 8;
    /// `flags`.
    pub const SUBMIT_FLAGS: usize = 12;
    /// `fence.id`, an `s32`.
    pub const SUBMIT_FENCE_ID: usize = 16;
    /// `fence.value`.
    pub const SUBMIT_FENCE_VALUE: usize = 20;

    /// `IoctlChannelSetTimeout`: 4 bytes.
    pub const TIMEOUT_BYTES: usize = 4;
    /// `timeout`.
    pub const TIMEOUT: usize = 0;

    /// `IoctlSetErrorNotifier`: 24 bytes.
    pub const ERROR_NOTIFIER_BYTES: usize = 24;
    /// `offset`, a `u64`.
    pub const NOTIFIER_OFFSET: usize = 0;
    /// `size`, a `u64`.
    pub const NOTIFIER_SIZE: usize = 8;
    /// `mem`, "nvmap object handle" (libnx calls the word `enable`); a padding word follows.
    pub const NOTIFIER_MEM: usize = 16;

    /// PR's `IoctlPreemptionModes`: 8 bytes (`static_assert(sizeof(IoctlPreemptionModes) ==
    /// 8)`).
    pub const PREEMPTION_BYTES: usize = 8;
    /// `graphics`.
    pub const PREEMPTION_GRAPHICS: usize = 0;
    /// `compute`.
    pub const PREEMPTION_COMPUTE: usize = 4;

    /// `IoctlClientData`: 8 bytes.
    pub const CLIENT_DATA_BYTES: usize = 8;
    /// `data`, a `u64`.
    pub const CLIENT_DATA: usize = 0;

    /// `IoctlZCullBind`: 16 bytes (`static_assert(sizeof(IoctlZCullBind) == 16)`).
    pub const ZCULL_BIND_BYTES: usize = 16;
    /// `gpu_va`, a `u64`.
    pub const ZCULL_GPU_VA: usize = 0;
    /// `mode` (PR's comment: `0=global, 1=no_ctxsw, 2=separate_buffer, 3=part_of_regular_buf`);
    /// a padding word follows.
    pub const ZCULL_MODE: usize = 8;
}

use layout::{
    ALLOC_FENCE_ID, ALLOC_FENCE_VALUE, ALLOC_GPFIFO_BYTES, CLIENT_DATA, CLIENT_DATA_BYTES,
    ERROR_NOTIFIER_BYTES, NOTIFIER_MEM, NOTIFIER_OFFSET, NOTIFIER_SIZE, OBJ_CLASS, OBJ_CTX_BYTES,
    OBJ_FLAGS, PREEMPTION_BYTES, PREEMPTION_COMPUTE, PREEMPTION_GRAPHICS, SUBMIT_BYTES,
    SUBMIT_FENCE_ID, SUBMIT_FENCE_VALUE, SUBMIT_FLAGS, SUBMIT_NUM_ENTRIES, TIMEOUT, TIMEOUT_BYTES,
    ZCULL_BIND_BYTES, ZCULL_GPU_VA, ZCULL_MODE,
};

/// `SubmitGPFIFO` flag bit 0: wait on the given fence first (`BitField<0, 1> fence_wait`).
pub const FENCE_WAIT: u32 = 1 << 0;
/// Bit 1: increment the channel's syncpoint by [`FENCE_INCREMENT_AMOUNT`] at the end
/// (`BitField<1, 1> fence_increment`).
pub const FENCE_INCREMENT: u32 = 1 << 1;
/// Bit 2: the new hardware format (`BitField<2, 1> new_hw_format`, "mostly ignored"): read by
/// nothing, here as in Eden's `SubmitGPFIFOImpl`.
pub const NEW_HW_FORMAT: u32 = 1 << 2;
/// Bit 4: skip the wait-for-idle (`BitField<4, 1> suppress_wfi`).
pub const SUPPRESS_WFI: u32 = 1 << 4;
/// Bit 8: increment the channel's syncpoint by the fence's value (`BitField<8, 1>
/// increment_value`).
pub const INCREMENT_VALUE: u32 = 1 << 8;

/// What bit 1 adds to the increment (`SubmitGPFIFOImpl`: `fence_increment ? 2 : 0`;
/// `BuildIncrementCommandList` pushes two increments).
pub const FENCE_INCREMENT_AMOUNT: u32 = 2;

/// The GPU front's queue bound, [`GpfifoEntry`], [`Submission`], [`Submitted`] and
/// [`ChannelConfig`]: graphics-server's shared types, defined in `crates/gpu` with the front that
/// enforces the bound (gpu-protocol slice G14) and re-exported here, where gsv's code names them.
pub use starquake_gpu::interface::{
    CHANNEL_QUEUE_ENTRIES, ChannelConfig, GpfifoEntry, Submission, Submitted,
};

/// The most bytes of GP entries a submission's request is read for: [`CHANNEL_QUEUE_ENTRIES`]
/// entries. The inline input of `SubmitGPFIFO` 0x1B is read up to this.
pub const MAX_ENTRY_BYTES: usize = CHANNEL_QUEUE_ENTRIES * GP_ENTRY_BYTES;

/// The channel part of the spec's `GpuFront` and the error it refuses with: graphics-server's
/// `ChannelFront` and `GpuError`, defined in `crates/gpu` with the front that implements them
/// (gpu-protocol slice G21, `starquake_gpu::interface`, "`GpuFront` in parts") and re-exported
/// here, where gsv's code names them. gsv defined them until the GPU crate had a front.
pub use starquake_gpu::interface::{ChannelFront, GpuError};

/// Eden's object-context classes (`nvhost_gpu.h`: `enum class CtxClasses`), the Maxwell engine
/// slots NX2 maps the Switch 2 classes onto.
pub mod class {
    /// `Ctx2D = 0x902D`.
    pub const CTX_2D: u32 = super::cl902d::CLASS;
    /// `Ctx3D = 0xB197`.
    pub const CTX_3D: u32 = 0xB197;
    /// `CtxCompute = 0xB1C0`.
    pub const CTX_COMPUTE: u32 = 0xB1C0;
    /// `CtxKepler = 0xA140` (inline to memory).
    pub const CTX_KEPLER: u32 = super::cla140::CLASS;
    /// `CtxDMA = 0xB0B5` (copy).
    pub const CTX_DMA: u32 = 0xB0B5;
    /// `CtxChannelGPFIFO = 0xB06F`.
    pub const CTX_CHANNEL_GPFIFO: u32 = 0xB06F;

    /// The slots in Eden's order (`GetObjectContextClassNumberIndex`: 2D 0, 3D 1, compute 2,
    /// Kepler 3, DMA 4, GPFIFO 5).
    pub const SLOTS: [u32; 6] = [
        CTX_2D,
        CTX_3D,
        CTX_COMPUTE,
        CTX_KEPLER,
        CTX_DMA,
        CTX_CHANNEL_GPFIFO,
    ];

    /// The slots class 0 may fill, in NX2's order (`for (const auto candidate : {Ctx3D,
    /// CtxCompute, CtxDMA, Ctx2D, CtxKepler})`).
    pub const ZERO_CANDIDATES: [u32; 5] = [CTX_3D, CTX_COMPUTE, CTX_DMA, CTX_2D, CTX_KEPLER];

    /// What NX2 maps onto 3D: `0xC797` (row 31), `0xC697`, `0xC597`.
    pub const ON_3D: [u32; 3] = [super::clc797::CLASS, 0xC697, 0xC597];
    /// What NX2 maps onto compute: `0xC7C0` (row 31), `0xC6C0`, `0xC5C0`.
    pub const ON_COMPUTE: [u32; 3] = [super::clc7c0::CLASS, 0xC6C0, 0xC5C0];
    /// What NX2 maps onto copy: `0xC7B5` (row 31), `0xC6B5`, `0xC5B5`.
    pub const ON_DMA: [u32; 3] = [super::clc7b5::CLASS, 0xC6B5, 0xC5B5];
    /// What NX2 maps onto the GPFIFO slot: `0xC56F`, `0xC76F`.
    pub const ON_GPFIFO: [u32; 2] = [0xC56F, super::clc76f::CLASS];
}

/// The number of engine slots ([`class::SLOTS`]).
pub const CLASS_SLOTS: usize = class::SLOTS.len();

/// The Maxwell class `requested` is mapped onto (NX2's `AllocateObjectContext` switch), or
/// itself; class 0 is not mapped here ([`resolve_class`]).
#[must_use]
pub fn slot_class(requested: u32) -> u32 {
    if class::ON_3D.contains(&requested) {
        class::CTX_3D
    } else if class::ON_COMPUTE.contains(&requested) {
        class::CTX_COMPUTE
    } else if class::ON_DMA.contains(&requested) {
        class::CTX_DMA
    } else if class::ON_GPFIFO.contains(&requested) {
        class::CTX_CHANNEL_GPFIFO
    } else {
        requested
    }
}

/// The engine slot of a mapped class (`GetObjectContextClassNumberIndex`), or `None` (Eden's
/// `-1`).
#[must_use]
pub fn slot_of(mapped: u32) -> Option<usize> {
    class::SLOTS.iter().position(|&slot| slot == mapped)
}

/// The class `requested` is mapped onto when `taken` slots are taken: [`slot_class`], or for
/// class 0 the first of [`class::ZERO_CANDIDATES`] whose slot is free, or 0 if none is.
#[must_use]
pub fn resolve_class(requested: u32, taken: &[usize]) -> u32 {
    if requested != 0 {
        return slot_class(requested);
    }
    class::ZERO_CANDIDATES
        .into_iter()
        .find(|&candidate| slot_of(candidate).is_some_and(|slot| !taken.contains(&slot)))
        .unwrap_or(0)
}

/// What a submission's flags and fence value make of it (module docs, "The ioctls"): whether it
/// waits, how many increments follow it, and whether a wait-for-idle comes before them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Flag bit 0: wait on the given fence first.
    pub wait: bool,
    /// The increment of the channel's syncpoint.
    pub increments: u32,
    /// Wait for idle first: flag bit 4 clear.
    pub wfi: bool,
}

/// The plan of a submission with `flags` and fence value `value`, or `BadParameter` for bits 0
/// and 8 together and for an increment past `u32::MAX`, in that order (module docs, "The ioctls",
/// checks 3 and 4).
pub fn plan(flags: u32, value: u32) -> Result<Plan, NvResult> {
    let wait = flags & FENCE_WAIT != 0;
    let by_value = flags & INCREMENT_VALUE != 0;
    if wait && by_value {
        return Err(NvResult::BadParameter);
    }
    let fixed = if flags & FENCE_INCREMENT == 0 {
        0
    } else {
        FENCE_INCREMENT_AMOUNT
    };
    let increments = fixed
        .checked_add(if by_value { value } else { 0 })
        .ok_or(NvResult::BadParameter)?;
    Ok(Plan {
        wait,
        increments,
        wfi: flags & SUPPRESS_WFI == 0,
    })
}

/// The events `QueryEvent` answers on an nvhost-gpu fd (module docs, "Events").
pub mod event {
    /// SM exception breakpoint interrupt (Eden `QueryEvent`: `case 1`).
    pub const SM_EXCEPTION_INT: u32 = 1;
    /// SM exception breakpoint pause (`case 2`).
    pub const SM_EXCEPTION_PAUSE: u32 = 2;
    /// Error notifier (`case 3`).
    pub const ERROR_NOTIFIER: u32 = 3;
    /// PR's `case 7`: the error notifier's event.
    pub const NATIVE_ERROR: u32 = 7;
}

/// How many distinct events a channel has: 1, 2 and 3 (7 is 3's).
pub const EVENT_SLOTS: usize = 3;

/// The slot of event `id`, or `None` for an id the device has no event for.
#[must_use]
pub const fn event_slot(id: u32) -> Option<usize> {
    match id {
        event::SM_EXCEPTION_INT => Some(0),
        event::SM_EXCEPTION_PAUSE => Some(1),
        event::ERROR_NOTIFIER | event::NATIVE_ERROR => Some(2),
        _ => None,
    }
}

/// A channel's GPU channel, once made, and the address space it was made in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Made {
    /// The front's channel id.
    pub channel: u32,
    /// The address space.
    pub space: AsId,
}

/// What `SetErrorNotifier` was given (`IoctlSetErrorNotifier`): kept, read by nothing yet
/// (module docs, "The settings").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorNotifier {
    /// `offset`.
    pub offset: u64,
    /// `size`.
    pub size: u64,
    /// `mem`, the nvmap handle the notifier lives in.
    pub mem: u32,
}

/// What `ZCullBind` was given (`IoctlZCullBind`): kept, read by nothing yet (module docs, "The
/// settings").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZcullBind {
    /// `gpu_va`: where the channel's zcull context buffer is.
    pub gpu_va: u64,
    /// `mode`, as sent: no reference checks it.
    pub mode: u32,
}

/// A channel's preemption modes (PR's `graphics_preemption_mode`, `compute_preemption_mode`):
/// kept, read by nothing yet. Each starts at [`preemption::DEFAULT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Preemption {
    /// The graphics mode.
    pub graphics: u32,
    /// The compute mode.
    pub compute: u32,
}

impl Default for Preemption {
    fn default() -> Self {
        Self {
            graphics: preemption::DEFAULT,
            compute: preemption::DEFAULT,
        }
    }
}

/// One nvhost-gpu fd's channel (module docs, "The channel").
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Channel {
    /// The syncpoint `AllocGPFIFOEx2` allocated.
    pub syncpoint: Option<u8>,
    /// The GPU channel, once made.
    pub made: Option<Made>,
    /// The classes given to the GPU, in the order `AllocObjCtx` bound them, one per slot.
    pub classes: Vec<u32>,
    /// The events made so far, by slot ([`event_slot`]).
    pub events: [Option<ObjectId>; EVENT_SLOTS],
    /// What `SetClientData` was given last, 0 before (PR `u64_le user_data{}`); `GetClientData`
    /// gives it back.
    pub client_data: u64,
    /// What `SetErrorNotifier` was given last, if it was called.
    pub error_notifier: Option<ErrorNotifier>,
    /// The modes `SetPreemptionModes` set.
    pub preemption: Preemption,
    /// What `ZCullBind` was given last, if it was called (the references' `zcull_params`, all
    /// zero until then; nothing reads it, so the two cannot be told apart).
    pub zcull: Option<ZcullBind>,
}

impl Channel {
    /// The slots its classes take.
    fn slots(&self) -> Vec<usize> {
        self.classes
            .iter()
            .filter_map(|&class| slot_of(resolve_class(class, &[])))
            .collect()
    }

    /// Its syncpoint, if it has one.
    fn syncpoint_id(&self) -> Option<SyncpointId> {
        self.syncpoint.and_then(SyncpointId::new)
    }
}

/// Whether a channel can hold preemption mode `mode` whose largest is `max`: the default, or,
/// once its GPU channel is made (`channel_made`), any mode `SetPreemptionModes` sets (from 1 to
/// `max`; module docs, "The settings").
const fn settable_mode(mode: u32, max: u32, channel_made: bool) -> bool {
    mode == preemption::DEFAULT || (channel_made && mode > preemption::KEEP && mode <= max)
}

/// Whether `id` is one `Syncpoints::allocate` can hand out: from [`FIRST_ALLOCATABLE`] up, below
/// the count, not reserved.
fn allocatable(id: u8) -> bool {
    id >= FIRST_ALLOCATABLE && SyncpointId::new(id).is_some() && !RESERVED_SYNCPOINTS.contains(&id)
}

/// Every nvhost-gpu fd's channel. Plain data, saved with the nvsrv state; deserialising goes
/// through `SavedNvhostGpu`, which refuses a state the ioctls cannot reach ([`GpuStateError`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedNvhostGpu")]
pub struct NvhostGpu {
    channels: BTreeMap<u32, Channel>,
}

/// The serialised form of [`NvhostGpu`].
#[derive(Deserialize)]
struct SavedNvhostGpu {
    channels: BTreeMap<u32, Channel>,
}

/// Why a saved [`NvhostGpu`] could not be read back. Deserialising checks, in this order, the
/// first that fails reported: each channel's syncpoint (fd order), then its GPU channel, then its
/// classes, then its preemption modes; then that no two channels share a syncpoint, then that no
/// two share a GPU channel.
/// [`NvhostGpu::check_syncpoints`] checks the last against the restored syncpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuStateError {
    /// A syncpoint `AllocGPFIFOEx2` cannot hand out.
    #[error("channel fd {fd} has syncpoint {id}, which a channel cannot be given")]
    Syncpoint {
        /// The channel's fd.
        fd: u32,
        /// The syncpoint.
        id: u8,
    },
    /// A GPU channel made with no syncpoint.
    #[error("channel fd {fd} has a GPU channel but no syncpoint")]
    MadeWithoutSyncpoint {
        /// The channel's fd.
        fd: u32,
    },
    /// A class `AllocObjCtx` does not bind, or two classes on one slot, or more than the slots.
    #[error("channel fd {fd} has class {class:#x}, which it cannot have")]
    Class {
        /// The channel's fd.
        fd: u32,
        /// The class.
        class: u32,
    },
    /// A preemption mode `SetPreemptionModes` cannot set: other than the default, unless the GPU
    /// channel is made and the mode is from 1 to its largest.
    #[error(
        "channel fd {fd} has graphics preemption mode {graphics} and compute mode {compute}, \
         which it cannot have"
    )]
    Preemption {
        /// The channel's fd.
        fd: u32,
        /// The graphics mode.
        graphics: u32,
        /// The compute mode.
        compute: u32,
    },
    /// Two channels with one syncpoint.
    #[error("channel fds {first} and {second} share syncpoint {id}")]
    SharedSyncpoint {
        /// The lower fd.
        first: u32,
        /// The higher fd.
        second: u32,
        /// The syncpoint.
        id: u8,
    },
    /// Two channels with one GPU channel.
    #[error("channel fds {first} and {second} share GPU channel {channel}")]
    SharedChannel {
        /// The lower fd.
        first: u32,
        /// The higher fd.
        second: u32,
        /// The GPU channel.
        channel: u32,
    },
    /// A channel's syncpoint the restored syncpoints do not have allocated
    /// ([`NvhostGpu::check_syncpoints`]).
    #[error("channel fd {fd} has syncpoint {id}, which is not allocated")]
    SyncpointNotAllocated {
        /// The channel's fd.
        fd: u32,
        /// The syncpoint.
        id: u8,
    },
}

impl TryFrom<SavedNvhostGpu> for NvhostGpu {
    type Error = GpuStateError;

    fn try_from(saved: SavedNvhostGpu) -> Result<Self, Self::Error> {
        let SavedNvhostGpu { channels } = saved;
        for (&fd, channel) in &channels {
            if let Some(id) = channel.syncpoint
                && !allocatable(id)
            {
                return Err(GpuStateError::Syncpoint { fd, id });
            }
            if channel.made.is_some() && channel.syncpoint.is_none() {
                return Err(GpuStateError::MadeWithoutSyncpoint { fd });
            }
            let mut slots = Vec::new();
            for &class in &channel.classes {
                let slot = slot_of(resolve_class(class, &[])).filter(|slot| !slots.contains(slot));
                let Some(slot) = slot.filter(|_| class != 0) else {
                    return Err(GpuStateError::Class { fd, class });
                };
                slots.push(slot);
            }
            let Preemption { graphics, compute } = channel.preemption;
            let made = channel.made.is_some();
            if !settable_mode(graphics, preemption::GRAPHICS_MAX, made)
                || !settable_mode(compute, preemption::COMPUTE_MAX, made)
            {
                return Err(GpuStateError::Preemption {
                    fd,
                    graphics,
                    compute,
                });
            }
        }
        let mut syncpoints: BTreeMap<u8, u32> = BTreeMap::new();
        for (&fd, channel) in &channels {
            if let Some(id) = channel.syncpoint
                && let Some(&first) = syncpoints.get(&id)
            {
                return Err(GpuStateError::SharedSyncpoint {
                    first,
                    second: fd,
                    id,
                });
            }
            if let Some(id) = channel.syncpoint {
                syncpoints.insert(id, fd);
            }
        }
        let mut made: BTreeMap<u32, u32> = BTreeMap::new();
        for (&fd, channel) in &channels {
            let Some(Made { channel: id, .. }) = channel.made else {
                continue;
            };
            if let Some(&first) = made.get(&id) {
                return Err(GpuStateError::SharedChannel {
                    first,
                    second: fd,
                    channel: id,
                });
            }
            made.insert(id, fd);
        }
        Ok(Self { channels })
    }
}

/// What nvhost-gpu reaches beside the nvsrv state: gsv's waiting submissions (under gsv's shared
/// lock), the syncpoints, the channel front, and where a submission that must wait is handed
/// back to the caller, which takes its reply token ([`super::NvsrvFamily::ioctl`]).
pub struct ChannelAccess<'a> {
    /// The submissions that wait for room.
    pub submits: &'a mut Submits,
    /// gsv's syncpoints.
    pub syncpoints: &'a Syncpoints,
    /// The channel front.
    pub front: &'a Arc<dyn ChannelFront>,
    /// Set by a submission that must wait.
    pub deferred: &'a mut Option<Deferral>,
}

/// A submission that must wait for room: its channel and what the front is to be given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deferral {
    /// The GPU channel.
    pub channel: ChannelId,
    /// The submission.
    pub submission: Submission,
}

/// What `QueryEvent` finds on an nvhost-gpu fd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Queried {
    /// The event, made already.
    Event(ObjectId),
    /// An event the device has but has not made: make it and [`NvhostGpu::set_event`] it.
    Make,
    /// No such event: `BadParameter`.
    Missing,
}

/// Where a submission's entries are: the bytes after the struct (`Ioctl`), or the inline input
/// (`Ioctl2`).
fn entry_bytes<'a>(call: &IoctlCall<'a>) -> &'a [u8] {
    match call.variant {
        IoctlVariant::Ioctl2 => call.inline_input,
        _ => call.input.get(SUBMIT_BYTES..).unwrap_or_default(),
    }
}

/// The whole 8-byte entries in `bytes`.
fn entries_of(bytes: &[u8]) -> Vec<GpfifoEntry> {
    bytes
        .as_chunks::<GP_ENTRY_BYTES>()
        .0
        .iter()
        .map(|entry| GpfifoEntry(u64::from_le_bytes(*entry)))
        .collect()
}

impl NvhostGpu {
    /// No channels.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The channel of fd `fd`, if an ioctl or event query made one.
    #[must_use]
    pub fn channel(&self, fd: u32) -> Option<&Channel> {
        self.channels.get(&fd)
    }

    /// The channels' fds, in order.
    pub fn fds(&self) -> impl Iterator<Item = u32> + '_ {
        self.channels.keys().copied()
    }

    /// The address space fd `fd`'s GPU channel was made in, if it was made.
    #[must_use]
    pub fn made_in(&self, fd: u32) -> Option<AsId> {
        self.channels.get(&fd)?.made.map(|made| made.space)
    }

    /// Whether every channel's syncpoint is allocated in `syncpoints`, in fd order, the first
    /// that is not reported. A restore that composes nvsrv's state with the restored syncpoints
    /// calls this (module docs, "What is recorded and what is saved").
    pub fn check_syncpoints(&self, syncpoints: &Syncpoints) -> Result<(), GpuStateError> {
        for (&fd, channel) in &self.channels {
            if let Some(id) = channel.syncpoint
                && !channel
                    .syncpoint_id()
                    .is_some_and(|id| syncpoints.is_allocated(id))
            {
                return Err(GpuStateError::SyncpointNotAllocated { fd, id });
            }
        }
        Ok(())
    }

    /// `QueryEvent` of event `id` on fd `fd` (module docs, "Events").
    #[must_use]
    pub fn event(&self, fd: u32, id: u32) -> Queried {
        let Some(slot) = event_slot(id) else {
            return Queried::Missing;
        };
        let made = self
            .channels
            .get(&fd)
            .and_then(|channel| channel.events.get(slot).copied().flatten());
        made.map_or(Queried::Make, Queried::Event)
    }

    /// Keeps `event` as fd `fd`'s event `id`, made for a [`Queried::Make`].
    pub(super) fn set_event(&mut self, fd: u32, id: u32, event: ObjectId) {
        if let Some(slot) = event_slot(id)
            && let Some(kept) = self.channels.entry(fd).or_default().events.get_mut(slot)
        {
            *kept = Some(event);
        }
    }

    /// Fd `fd` was closed: its waiting submissions are answered, its GPU channel destroyed and
    /// its syncpoint freed (module docs, "Where this differs"). Without `channels`
    /// (a server with no channel front) a channel holds only events.
    pub(super) fn close(&mut self, fd: u32, channels: Option<&mut ChannelAccess<'_>>) {
        let Some(channel) = self.channels.remove(&fd) else {
            return;
        };
        let Some(access) = channels else {
            if channel.syncpoint.is_some() {
                tracing::error!(fd, "nvsrv: a channel with a syncpoint closed with no front");
            }
            return;
        };
        if let Some(made) = channel.made {
            let id = ChannelId(made.channel);
            access.submits.cancel(id);
            access.front.destroy_channel(id);
        }
        if let Some(id) = channel.syncpoint_id()
            && let Err(error) = access.syncpoints.free(id)
        {
            tracing::error!(%error, fd, "nvsrv: a closed channel's syncpoint was not allocated");
        }
    }

    /// An ioctl on nvhost-gpu fd `fd`, or `None` for one the device does not know (module docs,
    /// "The ioctls"). `as_gpu` says which address space the channel is bound to.
    pub(super) fn ioctl(
        &mut self,
        fd: u32,
        call: &IoctlCall<'_>,
        as_gpu: &AsGpu,
        access: &mut ChannelAccess<'_>,
    ) -> Option<IoctlAnswer> {
        let ioctl = call.ioctl;
        if ioctl.group() == CLIENT_GROUP {
            return match (call.variant, ioctl.command()) {
                (IoctlVariant::Ioctl, client_command::SET_CLIENT_DATA) => {
                    Some(self.set_client_data(fd, call))
                }
                (IoctlVariant::Ioctl, client_command::GET_CLIENT_DATA) => {
                    Some(self.get_client_data(fd, call))
                }
                _ => None,
            };
        }
        if ioctl.group() != GROUP {
            return None;
        }
        match (call.variant, ioctl.command()) {
            (IoctlVariant::Ioctl, command::ALLOC_GPFIFO_EX2) => {
                let mut arg = Fixed::<ALLOC_GPFIFO_BYTES>::from_input(call.input);
                let result = self.alloc_gpfifo(fd, &mut arg, as_gpu, access);
                Some(IoctlAnswer::with(result, arg.into_output()))
            }
            (IoctlVariant::Ioctl, command::ALLOC_OBJ_CTX) => {
                let mut arg = Fixed::<OBJ_CTX_BYTES>::from_input(call.input);
                let result = self.alloc_obj_ctx(fd, &mut arg, access);
                Some(IoctlAnswer::with(result, arg.into_output()))
            }
            (IoctlVariant::Ioctl, command::SUBMIT_GPFIFO)
            | (IoctlVariant::Ioctl2, command::SUBMIT_GPFIFO_INLINE) => {
                Some(self.submit(fd, call, as_gpu, access))
            }
            (IoctlVariant::Ioctl, command::CHANNEL_SET_TIMEOUT) => Some(set_timeout(fd, call)),
            (IoctlVariant::Ioctl, command::SET_ERROR_NOTIFIER) => {
                Some(self.set_error_notifier(fd, call))
            }
            (IoctlVariant::Ioctl, command::SET_PREEMPTION_MODES)
                if ioctl.raw() == SET_PREEMPTION_MODES =>
            {
                Some(self.set_preemption_modes(fd, call))
            }
            (IoctlVariant::Ioctl, command::ZCULL_BIND) => Some(self.zcull_bind(fd, call)),
            _ => None,
        }
    }

    /// `SetErrorNotifier` (module docs, "The settings"): kept, `Success`, the struct back.
    fn set_error_notifier(&mut self, fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
        let arg = Fixed::<ERROR_NOTIFIER_BYTES>::from_input(call.input);
        let notifier = ErrorNotifier {
            offset: arg.u64_at::<NOTIFIER_OFFSET>(),
            size: arg.u64_at::<NOTIFIER_SIZE>(),
            mem: arg.u32_at::<NOTIFIER_MEM>(),
        };
        tracing::debug!(
            fd,
            offset = notifier.offset,
            size = notifier.size,
            mem = notifier.mem,
            "nvsrv: a channel's error notifier"
        );
        self.channels.entry(fd).or_default().error_notifier = Some(notifier);
        IoctlAnswer::with(NvResult::Success, arg.into_output())
    }

    /// `SetPreemptionModes` (module docs, "The settings"): the struct back whatever the result.
    fn set_preemption_modes(&mut self, fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
        let arg = Fixed::<PREEMPTION_BYTES>::from_input(call.input);
        let graphics = arg.u32_at::<PREEMPTION_GRAPHICS>();
        let compute = arg.u32_at::<PREEMPTION_COMPUTE>();
        let result = match self.channels.get_mut(&fd) {
            Some(channel) if channel.made.is_some() => {
                if graphics > preemption::GRAPHICS_MAX || compute > preemption::COMPUTE_MAX {
                    NvResult::NotSupported
                } else {
                    if graphics != preemption::KEEP {
                        channel.preemption.graphics = graphics;
                    }
                    if compute != preemption::KEEP {
                        channel.preemption.compute = compute;
                    }
                    NvResult::Success
                }
            }
            _ => NvResult::InvalidState,
        };
        if result != NvResult::Success {
            tracing::warn!(
                fd,
                graphics,
                compute,
                result = result.raw(),
                "nvsrv: preemption modes refused"
            );
        }
        IoctlAnswer::with(result, arg.into_output())
    }

    /// `ZCullBind` (module docs, "The settings"): kept, `Success`, the struct back (PR:
    /// `zcull_params = params; .. return NvResult::Success;`).
    fn zcull_bind(&mut self, fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
        let arg = Fixed::<ZCULL_BIND_BYTES>::from_input(call.input);
        let zcull = ZcullBind {
            gpu_va: arg.u64_at::<ZCULL_GPU_VA>(),
            mode: arg.u32_at::<ZCULL_MODE>(),
        };
        tracing::debug!(
            fd,
            gpu_va = zcull.gpu_va,
            mode = zcull.mode,
            "nvsrv: a channel's zcull bind"
        );
        self.channels.entry(fd).or_default().zcull = Some(zcull);
        IoctlAnswer::with(NvResult::Success, arg.into_output())
    }

    /// `SetClientData` (module docs, "The settings"): kept, `Success`, the struct back.
    fn set_client_data(&mut self, fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
        let arg = Fixed::<CLIENT_DATA_BYTES>::from_input(call.input);
        self.channels.entry(fd).or_default().client_data = arg.u64_at::<CLIENT_DATA>();
        IoctlAnswer::with(NvResult::Success, arg.into_output())
    }

    /// `GetClientData` (module docs, "The settings"): the kept data, 0 if none, `Success`.
    fn get_client_data(&self, fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
        let mut arg = Fixed::<CLIENT_DATA_BYTES>::from_input(call.input);
        let data = self
            .channels
            .get(&fd)
            .map_or(0, |channel| channel.client_data);
        arg.set_u64::<CLIENT_DATA>(data);
        IoctlAnswer::with(NvResult::Success, arg.into_output())
    }

    /// `AllocGPFIFOEx2` (module docs, "The ioctls").
    fn alloc_gpfifo(
        &mut self,
        fd: u32,
        arg: &mut Fixed<ALLOC_GPFIFO_BYTES>,
        as_gpu: &AsGpu,
        access: &mut ChannelAccess<'_>,
    ) -> NvResult {
        let mut channel = self.channels.get(&fd).cloned().unwrap_or_default();
        if channel.syncpoint.is_some() {
            return NvResult::AlreadyAllocated;
        }
        let id = match access.syncpoints.allocate() {
            Ok(id) => id,
            Err(error) => {
                tracing::warn!(%error, fd, "nvsrv: no syncpoint for a channel");
                return NvResult::InsufficientMemory;
            }
        };
        channel.syncpoint = Some(id.get());
        if let Some(space) = as_gpu.bound(fd)
            && let Err(result) = make(&mut channel, space, access)
        {
            if let Err(error) = access.syncpoints.free(id) {
                tracing::error!(%error, fd, "nvsrv: a syncpoint just allocated was not");
            }
            return result;
        }
        let Ok(fence) = access.syncpoints.fence(id) else {
            // `allocate` just handed it out.
            return NvResult::InvalidState;
        };
        arg.set_u32::<ALLOC_FENCE_ID>(u32::from(id.get()));
        arg.set_u32::<ALLOC_FENCE_VALUE>(fence.value);
        self.channels.insert(fd, channel);
        NvResult::Success
    }

    /// `AllocObjCtx` (module docs, "The ioctls").
    fn alloc_obj_ctx(
        &mut self,
        fd: u32,
        arg: &mut Fixed<OBJ_CTX_BYTES>,
        access: &ChannelAccess<'_>,
    ) -> NvResult {
        let requested = arg.u32_at::<OBJ_CLASS>();
        let flags = arg.u32_at::<OBJ_FLAGS>();
        if flags != 0 {
            tracing::warn!(flags, requested, "nvsrv: object context flags ignored");
            arg.set_u32::<OBJ_FLAGS>(0);
        }
        let channel = self.channels.get(&fd);
        let taken = channel.map_or_default(Channel::slots);
        let mapped = resolve_class(requested, &taken);
        arg.set_u32::<OBJ_CLASS>(mapped);
        let Some(slot) = slot_of(mapped) else {
            return NvResult::BadParameter;
        };
        if taken.contains(&slot) {
            return NvResult::AlreadyAllocated;
        }
        let given = if requested == 0 { mapped } else { requested };
        if let Some(made) = channel.and_then(|channel| channel.made)
            && let Err(error) = access.front.bind_class(ChannelId(made.channel), given)
        {
            tracing::warn!(%error, class = given, "nvsrv: the GPU refused an object class");
            return NvResult::BadParameter;
        }
        self.channels.entry(fd).or_default().classes.push(given);
        NvResult::Success
    }

    /// `SubmitGPFIFO` through 0x8 or 0x1B (module docs, "The ioctls"): the argument given back,
    /// and for 0x8 the entries after it.
    fn submit(
        &mut self,
        fd: u32,
        call: &IoctlCall<'_>,
        as_gpu: &AsGpu,
        access: &mut ChannelAccess<'_>,
    ) -> IoctlAnswer {
        let mut arg = Fixed::<SUBMIT_BYTES>::from_input(call.input);
        let source = entry_bytes(call);
        let echo = |arg: Fixed<SUBMIT_BYTES>| {
            let mut output = arg.into_output();
            if call.variant == IoctlVariant::Ioctl {
                let whole = source.len() / GP_ENTRY_BYTES * GP_ENTRY_BYTES;
                output.extend(source.get(..whole).unwrap_or_default());
            }
            output
        };
        match self.submission(fd, &arg, source, as_gpu, access) {
            Err(result) => IoctlAnswer::with(result, echo(arg)),
            Ok((channel, fence, submission)) => {
                arg.set_u32::<SUBMIT_FENCE_ID>(u32::from(fence.id.get()));
                arg.set_u32::<SUBMIT_FENCE_VALUE>(fence.value);
                arg.set_u32::<SUBMIT_FLAGS>(0);
                if access.submits.is_waiting(channel)
                    || access.front.submit(channel, submission.clone()) == Submitted::Full
                {
                    *access.deferred = Some(Deferral {
                        channel,
                        submission,
                    });
                }
                IoctlAnswer::with(NvResult::Success, echo(arg))
            }
        }
    }

    /// The checks of `SubmitGPFIFO` and its fence (module docs, "The ioctls"): the GPU channel,
    /// the fence the reply gives back, and what the front is to be given. The channel's
    /// syncpoint's maximum is raised only once every check has passed.
    fn submission(
        &mut self,
        fd: u32,
        arg: &Fixed<SUBMIT_BYTES>,
        source: &[u8],
        as_gpu: &AsGpu,
        access: &mut ChannelAccess<'_>,
    ) -> Result<(ChannelId, Fence, Submission), NvResult> {
        let count = usize::try_from(arg.u32_at::<SUBMIT_NUM_ENTRIES>()).unwrap_or(usize::MAX);
        if count > source.len() / GP_ENTRY_BYTES || count > CHANNEL_QUEUE_ENTRIES {
            return Err(NvResult::InvalidSize);
        }
        let plan = plan(
            arg.u32_at::<SUBMIT_FLAGS>(),
            arg.u32_at::<SUBMIT_FENCE_VALUE>(),
        )?;
        let wait = if plan.wait {
            let id = u8::try_from(arg.u32_at::<SUBMIT_FENCE_ID>().cast_signed())
                .ok()
                .and_then(SyncpointId::new)
                .ok_or(NvResult::BadParameter)?;
            Some(Fence {
                id,
                value: arg.u32_at::<SUBMIT_FENCE_VALUE>(),
            })
        } else {
            None
        };
        let channel = self.channels.get_mut(&fd).ok_or(NvResult::InvalidState)?;
        let syncpoint = channel.syncpoint_id().ok_or(NvResult::InvalidState)?;
        let id = if let Some(made) = channel.made {
            ChannelId(made.channel)
        } else {
            let space = as_gpu.bound(fd).ok_or(NvResult::InvalidState)?;
            make(channel, space, access)?
        };
        // Check 7: before the maximum is raised, so no fence goes out for a refused one.
        if !access.submits.admits(id, count) {
            return Err(NvResult::Busy);
        }
        let value = access
            .syncpoints
            .increment_max(syncpoint, plan.increments)
            .map_err(|error| {
                tracing::error!(%error, fd, "nvsrv: a channel's syncpoint was not allocated");
                NvResult::InvalidState
            })?;
        let entries = entries_of(source.get(..count * GP_ENTRY_BYTES).unwrap_or_default());
        let submission = Submission {
            entries,
            wait,
            increments: plan.increments,
            wfi: plan.wfi,
        };
        let fence = Fence {
            id: syncpoint,
            value,
        };
        Ok((id, fence, submission))
    }
}

/// `ChannelSetTimeout` (module docs, "The settings"): logged, nothing kept, `Success`, the struct
/// back (PR: `LOG_INFO(.., "called, timeout={:#x}", params.timeout); return
/// NvResult::Success;`).
fn set_timeout(fd: u32, call: &IoctlCall<'_>) -> IoctlAnswer {
    let arg = Fixed::<TIMEOUT_BYTES>::from_input(call.input);
    tracing::info!(
        fd,
        timeout = arg.u32_at::<TIMEOUT>(),
        "nvsrv: a channel's timeout"
    );
    IoctlAnswer::with(NvResult::Success, arg.into_output())
}

/// Makes `channel`'s GPU channel in `space` and binds its classes on it, in the order bound; a
/// class the front refuses is logged. `InvalidState` without a syncpoint, `InsufficientMemory`
/// if the front refuses the channel (a guess).
fn make(
    channel: &mut Channel,
    space: AsId,
    access: &ChannelAccess<'_>,
) -> Result<ChannelId, NvResult> {
    let syncpoint = channel.syncpoint_id().ok_or(NvResult::InvalidState)?;
    let id = access
        .front
        .create_channel(ChannelConfig { space, syncpoint })
        .map_err(|error| {
            tracing::warn!(%error, "nvsrv: the GPU refused a channel");
            NvResult::InsufficientMemory
        })?;
    for &class in &channel.classes {
        if let Err(error) = access.front.bind_class(id, class) {
            tracing::warn!(%error, class, "nvsrv: the GPU refused an object class");
        }
    }
    channel.made = Some(Made {
        channel: id.0,
        space,
    });
    Ok(id)
}

/// Whether nvhost-gpu reads the inline input of `ioctl` sent through `variant`: `SubmitGPFIFO`
/// 0x1B on `Ioctl2`, whose entries are there.
#[must_use]
pub fn reads_inline_input(variant: IoctlVariant, ioctl: Ioctl) -> bool {
    variant == IoctlVariant::Ioctl2
        && ioctl.group() == GROUP
        && ioctl.command() == command::SUBMIT_GPFIFO_INLINE
}
