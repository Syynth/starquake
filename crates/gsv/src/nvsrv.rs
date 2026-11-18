//! nvsrv: the NVIDIA driver session gsv 1402 returns, which speaks the old nvdrv
//! `INvDrvServices` protocol (`Open`, the three `Ioctl` variants, `Close`, `QueryEvent`) over the devices
//! Switch 2's NvRm opens (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv";
//! port slices 9 and 10).
//!
//! * [`ioctl`]: ioctl numbers, and the fixed-size argument structs devices read.
//! * [`nvmap`]: `/dev/nvmap` and its handle table.
//! * [`ctrl_gpu`]: `/dev/nvhost-ctrl-gpu`, the GPU's characteristics block, its layout replies
//!   and its two events (slice 10).
//! * [`as_gpu`]: `/dev/nvhost-as-gpu`, address-space views over gsv's GPU memory (slice 10).
//! * [`tsg_gpu`]: `/dev/nvhost-tsg-gpu`, timeslice groups (slice 10).
//! * [`gpu`]: `/dev/nvhost-gpu`, channels and GPFIFO submission (slice 11).
//!
//! Ported from Eden (GPL-3.0-or-later) @ cf654052: `eden/src/core/hle/service/nvdrv/`
//! `nvdrv_interface.cpp` (`NVDRV`: the commands and their framing), `nvdrv.cpp` (`Module`: the
//! device names, the open-file table and its checks), `nvdata.h` (`NvResult`, `DeviceFD`) and
//! `devices/nvdevice.h` (what a device answers by default). "Row N" is a row of NX2's blocker log
//! (`eden-nx2/NX2_NOTES.md` @ 50fa86bb), the rows `docs/switch2-eden-diff-scoping.md` cites.
//!
//! # The protocol
//!
//! nvsrv is the nvdrv protocol, opened without `Initialize` (row 23, observed). Both Switch 2
//! references serve it with Eden's own `NVDRV` class: NX2 initialises it for the calling process
//! when 1402 makes it (`eden-nx2/src/core/hle/service/nx2_gsv.h`, `OpenNvsrvSession`;
//! `nvdrv_interface.cpp`, `InitializeForProcess`), PR in its constructor
//! (`eden-pr4588/src/core/hle/service/nvdrv/nvdrv_interface.cpp` @ e5fb3b97, the `NVDRV`
//! constructor that takes a process; `graphics_supervisor.cpp`, `GetNvidiaDriver`). So a session
//! here is initialised from the start, and no command answers `NotInitialized`.
//!
//! **CMIF, not TIPC.** Both references register the commands as CMIF (`RegisterHandlers`), and
//! NvRm's ioctls reached them (rows 24 to 26 are fixes to those ioctls), so CMIF is what titles
//! ran on. Which framing the console's nvsrv uses is still the spec's open question 9 ("What to
//! verify before building"); a TIPC request goes to the server's unknown-command policy
//! (`a_tipc_request_goes_to_the_policy`).
//!
//! Every reply is a CMIF result of success and, after it, the data below; what went wrong is
//! the [`NvResult`] word (`NVDRV`: `rb.Push(ResultSuccess); rb.PushEnum(nv_result)`).
//!
//! | Id | Command | In | Out |
//! |---|---|---|---|
//! | 0 | `Open` | in buffer 0 (map-alias): the device's path | `s32` fd, [`NvResult`] |
//! | 1 | `Ioctl` | `s32` fd, `u32` ioctl; in buffer 0 (auto-select): the argument | out buffer 0 (auto-select): the argument back; [`NvResult`] |
//! | 2 | `Close` | `s32` fd | [`NvResult`] |
//! | 3 | `Initialize` | nothing read | [`NvResult`] `Success` |
//! | 4 | `QueryEvent` | `s32` fd, `u32` event id | [`NvResult`] |
//! | 6 | `GetStatus` | nothing | [`NvResult`] `Success` |
//! | 8 | `SetAruid` | `u64` applet resource user id | [`NvResult`] `Success` |
//! | 9 | `DumpGraphicsMemoryInfo` | nothing | nothing |
//! | 11 | `Ioctl2` | as `Ioctl`, and in buffer 1 (auto-select): inline input | as `Ioctl` |
//! | 12 | `Ioctl3` | as `Ioctl` | as `Ioctl`, and out buffer 1 (auto-select): inline output |
//! | 13 | `SetGraphicsFirmwareMemoryMarginEnabled` | nothing read | nothing |
//!
//! The ids are Eden's handler table (`NVDRV::NVDRV`); NX2's comment on 1402 names 13 the same.
//! The buffers are libnx's (`libnx/nx/source/services/nv.c` @ b156b477: `nvOpen` sends the path
//! as `SfBufferAttr_In | SfBufferAttr_HipcMapAlias`, and `nvIoctl`, `nvIoctl2` and `nvIoctl3`
//! send every buffer as `SfBufferAttr_HipcAutoSelect`), which Eden reads as it finds them
//! (`ReadBuffer`, `WriteBuffer`). Eden lists 5 (`MapSharedMem`), 7 (`SetAruidForTest`) and 10
//! (`InitializeDevtools`) with no handler, so they, and every id not in the table, go to the
//! server's unknown-command policy.
//!
//! `Initialize` answers at once: Eden's returns before it reads anything when the session is
//! initialised (`if (is_initialized) return;`), which an nvsrv session always is. `GetStatus`,
//! `SetAruid`, `DumpGraphicsMemoryInfo` and 13 are Eden's stubs, reply for reply.
//!
//! # Devices and fds
//!
//! `Open` names a device by path ([`Device`]): every path Eden's `Module` builds a device for,
//! and `/dev/nvhost-tsg-gpu`, which PR adds (`eden-pr4588/src/core/hle/service/nvdrv/nvdrv.cpp`)
//! and Switch 2's NvRm opens (rows 26, 45). PR gates that one device: its `Module::Open` returns
//! `INVALID_NVDRV_FD` for `/dev/nvhost-tsg-gpu` unless the session's process
//! `UsesPAuthReturns()`, which PR sets from the title's metadata
//! (`eden-pr4588/src/core/hle/kernel/k_process.cpp`: `m_uses_pauth_returns =
//! metadata.UsesExtendedAcid()`). Here it opens with no such check: nvsrv is reachable only
//! through gsv 1402, which only Switch 2 processes call, so every caller is one the gate lets
//! through. That every Switch 2 title's metadata has the extended ACID is a guess; a title that
//! reaches 1402 without it would confirm or refute it. The path is the buffer's bytes up to the
//! first zero byte (`Common::StringFromBuffer`). It answers:
//!
//! - `/dev/nvhost-prof-gpu` ([`PROFILER_PATH`]): fd 0 and `NotSupported` ("cannot be opened in
//!   production"), before anything else is looked at;
//! - any other path no device has: fd -1 ([`INVALID_FD`]) and `FileOperationFailed`;
//! - otherwise the next fd, counting up from 1 and never reused (`Module::next_fd`), and
//!   `Success`.
//!
//! The open files are the family's ([`NvsrvState`]): in both references every nvsrv session
//! shares the one nvdrv `Module`, its fds and its nvmap handles. Closing an IPC session closes
//! none of its fds, as in Eden, where they belong to the module. Closing the server's last
//! session frees the process's nvmap handles, as Eden's `Container::CloseSession` does when the
//! process's last nvdrv session goes ([`nvmap`], "When the process's last session closes"); ending
//! the process closes every session it holds, so that is also what its end does.
//!
//! An fd a command names is checked as Eden's `Module` checks it: negative → `InvalidState`, not
//! open → `NotImplemented`, before the device is asked anything.
//!
//! **What each device does.** `/dev/nvmap` has its ioctls ([`nvmap`]); `/dev/nvhost-ctrl-gpu`
//! answers `GetCharacteristics` and the GPU's layout (Z-cull, TPC masks, the SM map), and has two
//! events ([`ctrl_gpu`]); `/dev/nvhost-as-gpu` makes views of gsv's GPU
//! address spaces and reports their regions and syncpoint shim ([`as_gpu`]); and
//! `/dev/nvhost-tsg-gpu` keeps timeslice groups ([`tsg_gpu`]); and `/dev/nvhost-gpu` is a
//! channel ([`gpu`]), on a server given a channel front ([`NvsrvFamily::with_channel_front`]:
//! without one its ioctls, and as-gpu's `BindChannel`, go to the unknown-ioctl policy). Each lists
//! the ioctls it does not serve yet. The others open, but have no ioctls: they are Switch 1
//! devices (decision 11 builds only what Switch 2's driver uses).
//!
//! **ctrl-gpu's events.** Opening `/dev/nvhost-ctrl-gpu` makes its two events first, as both
//! references' `nvhost_ctrl_gpu` constructor does ([`ctrl_gpu`], "Events"): `Open` asks the
//! kernel for two events, then opens the fd with them ([`NvsrvState::opens_ctrl_gpu`],
//! [`NvsrvState::open`]). A kernel that cannot make one fails the request with its error, and no
//! fd is opened (an event made before the failure is not freed: the services framework has no
//! call for it). A path `Open` refuses makes no event.
//!
//! `QueryEvent` on an nvhost-gpu fd answers its channel's events ([`gpu`], "Events"), and on an
//! nvhost-ctrl-gpu fd its two ([`ctrl_gpu`], "Events"): a copied handle and `Success` (Eden
//! `NVDRV::QueryEvent`: `rb.PushCopyObjects(ctx, readable_event);
//! rb.PushEnum(NvResult::Success)`). An id that device has no event for, and any id on another
//! open fd, answers `BadParameter` and no handle, Eden's answer for a device with no such event
//! (`nvdevice::QueryEvent` returns `nullptr`).
//!
//! **The server's wiring.** nvhost-as-gpu acts on gsv's GPU memory, the bookkeeping the gpu
//! session's 33 and its worker use, so [`nvsrv_server`] takes gsv's shared state and the GPU's
//! front as [`crate::gpu::gpu_server`] does, and its family ([`NvsrvFamily`]) holds them beside
//! the state. Every ioctl and every `Close` runs with gsv's shared lock held
//! ([`NvsrvFamily::ioctl`], [`NvsrvFamily::close`]: closing an as-gpu fd can destroy an address
//! space, [`as_gpu`], "Closing"); nothing under it calls the kernel, and the front never calls
//! back into gsv. nvmap's device-share lock and release are taken through the kernel after the
//! ioctl, with gsv's lock released ([`nvmap`], "The device-share lock").
//!
//! # Ioctls
//!
//! `Ioctl`, `Ioctl2` and `Ioctl3` read the fd and the ioctl number, read the in buffer, check
//! the fd, and hand the device an [`IoctlCall`]: the ioctl ([`ioctl::Ioctl`]), the argument, and
//! the sizes of out buffer 0 and, for `Ioctl3`, out buffer 1 (which ctrl-gpu and tsg-gpu check).
//! If the ioctl's [`is_out`](ioctl::Ioctl::is_out) bit is set, what the device gives back is
//! written to out buffer 0, clipped to the buffer (`WrapGeneric`: `std::min(output.size(),
//! sizeof(FixedArg))`), whatever the result; and for `Ioctl3`, what it gives back inline is
//! written to out buffer 1, clipped likewise. A device gives back a fixed struct whatever the
//! result, as Eden's `WrapGeneric` does (nvmap, as-gpu), and nothing on a refusal where PR
//! returns before filling its buffers (ctrl-gpu's characteristics, tsg-gpu).
//!
//! **An ioctl the device does not know** (`docs/specs/graphics-server.md`: "not implemented" under
//! the strict policy and a zeroed success under auto-stub), by the server's [`UnknownPolicy`]:
//!
//! - strict: `NotImplemented`, the device's own answer in Eden (`UNIMPLEMENTED_MSG` and
//!   `return NvResult::NotImplemented`), logged, and nothing written;
//! - auto-stub: `Success`, logged, and if `is_out` is set, zeros written to out buffer 0, as many
//!   as the ioctl's declared [`size`](ioctl::Ioctl::size), clipped to the buffer. Nothing of the
//!   request is echoed: NX2's stub echoed the input (`nx2_stub_device.h`), which the service
//!   framework forbids since row 96.
//!
//! The fd checks come first under both policies: a bad fd is never auto-stubbed
//! (`fd_checks_come_before_the_policy`).
//!
//! # What is recorded and what is saved
//!
//! Every request is one IPC event in the run's stream, recorded by the service framework with
//! its payload; nvsrv has no input from the host, so it writes no notes. The open files, the
//! nvmap handles, nvhost-as-gpu's views and the layouts of the address spaces it made,
//! nvhost-tsg-gpu's groups, nvhost-gpu's channels and nvhost-ctrl-gpu's events are the family
//! state, plain data saved with the server (the wiring is not saved), and restoring checks every
//! relation between their fields ([`NvsrvStateError`], [`nvmap::NvMapStateError`],
//! [`as_gpu::AsGpuStateError`], [`tsg_gpu::TsgStateError`], [`gpu::GpuStateError`],
//! [`ctrl_gpu::CtrlGpuStateError`]): each view on an open as-gpu fd, each group on an open
//! tsg-gpu fd, each channel a group holds one whose fd was handed out and, while it is still
//! open, an nvhost-gpu, each nvhost-gpu channel and each as-gpu binding on an open nvhost-gpu fd,
//! each made GPU channel bound to the address space it was made in, each fd with ctrl-gpu events
//! an open ctrl-gpu, and each open ctrl-gpu with its events (in that order, after the fd
//! checks). The submissions that wait for room are gsv's shared state's ([`gpu::submits`]).
//!
//! The address spaces themselves are gsv's shared state's, saved with it
//! ([`crate::gpu::saved`]). That a view's address space is one the shared state holds is a
//! relation between the two snapshots, which neither restore can check alone: composing them is
//! the wiring slice's (16), as for the gpu session's, which checks it with
//! [`as_gpu::AsGpu::check_spaces`]. Until then a view whose address space is
//! missing answers its shim with `BadValue` ([`as_gpu`], `shim_page`) and nothing panics.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `nvsrv/tests.rs`:
//!
//! | Case | Eden (and NX2, PR) | Here |
//! |---|---|---|
//! | An in buffer larger than [`ioctl::MAX_ARGUMENT_BYTES`] | read whole | its first [`ioctl::MAX_ARGUMENT_BYTES`] bytes are read, the rest never touched: only nvhost-gpu's `SubmitGPFIFO` 0x8 reads past its fixed struct (its entries, as many whole ones as those bytes hold), and a title cannot make the server copy more (`an_in_buffer_is_read_up_to_the_largest_argument`; `gpu/tests.rs`: `the_count_is_checked_against_the_entries_sent`) |
//! | An out buffer, written when `is_out` is set | written whole, past what the device gave back too, with whatever the server's scratch buffer held; also on an fd error | written only as far as the device's answer (or the auto-stub's zeros); nothing on an fd error or a strict unknown ioctl (`an_out_buffer_gets_only_the_answer`, `fd_checks_come_before_the_policy`) |
//! | `Ioctl2`'s inline input | read | read only for nvhost-gpu's `SubmitGPFIFO` 0x1B on a server with a channel front, up to [`gpu::MAX_ENTRY_BYTES`] (a submission's entries can be no more); otherwise not read, also on another fd of a server with a channel front (`inline_buffers_are_left_alone`; `gpu/tests.rs`: `inline_input_is_read_only_on_a_channel_fd`, `a_submission_larger_than_a_queue_is_refused`) |
//! | `Ioctl3`'s inline output, written when `is_out` is set | written whole, with whatever the server's scratch buffer held past the device's answer | written only as far as the device's answer, nothing when it gives none (`inline_buffers_are_left_alone`; `ctrl_gpu/tests.rs`: `only_the_packet_and_the_block_are_written`) |
//! | A path no device has, under auto-stub | NX2 opens any `/dev/nvhost-*` path as a stub device | refused under both policies, as Eden refuses it (`an_unknown_path_is_refused_under_both_policies`) |
//! | [`MAX_OPEN_FILES`] files open, or fds used up | no limit; the next fd overflows its `s32` | fd -1 and `FileOperationFailed` (a guess), nothing opened and no fd used (`open_refuses_a_full_table_and_spent_fds`) |
//! | A TIPC request | NX2 and PR: auto-stubbed or refused, as for any command they lack | the server's unknown-command policy (`a_tipc_request_goes_to_the_policy`) |
//! | An ioctl the device does not know, under auto-stub | the spec asks for the service framework's auto-stub policy (`docs/specs/services.md`): logged once per command, and the request marked as stubbed in the trace | answered as the spec says, but the request is recorded as `Done`, not `Stubbed`, and is not listed in `Server::stubbed`: the framework marks only the commands it stubs itself, and `Ioctl` is a command nvsrv serves. Every such call is logged, at warn, not only the first (`an_auto_stubbed_ioctl_is_recorded_as_done`) |
//!
//! A change an ioctl made stands if writing its out buffer then fails (the caller's memory is not
//! mapped): the caller gets the kernel's error, as for any service whose reply writes a buffer.
//! In Eden the write is dropped and the change stands too. For a submission that must wait for
//! room ([`gpu`]), the out buffer is written before the reply token is taken: if that write
//! fails, the request fails with the kernel's error, the submission never waits, and the
//! channel syncpoint's maximum stays raised by its increment.

pub mod as_gpu;
pub mod ctrl_gpu;
pub mod gpu;
pub mod ioctl;
pub mod nvmap;
pub mod tsg_gpu;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, NOT_SUPPORTED, Outcome, Reply, ReplyToken, Request, Server,
    SnapshotError, UnknownPolicy,
};

use self::as_gpu::AsGpu;
use self::ctrl_gpu::{CtrlGpu, CtrlGpuEvents};
use self::gpu::{ChannelAccess, ChannelFront, Deferral, NvhostGpu, Queried};
use self::ioctl::{Ioctl, MAX_ARGUMENT_BYTES};
use self::nvmap::{DeviceShareStep, NvMap};
use self::tsg_gpu::TsgGpu;
use crate::completion::Completion;
use crate::gpu::{GpuMemory, GpuVmFront};
use crate::shared::Shared;

#[cfg(test)]
mod tests;

/// The nvsrv server's name, in logs. It is not a registry name: titles reach nvsrv only through
/// gsv 1402.
pub const NVSRV_NAME: &str = "nvsrv";

/// The nvdrv command ids (Eden `NVDRV::NVDRV`'s handler table).
pub mod command {
    /// `{0, &NVDRV::Open, "Open"}`.
    pub const OPEN: u32 = 0;
    /// `{1, &NVDRV::Ioctl1, "Ioctl"}`.
    pub const IOCTL: u32 = 1;
    /// `{2, &NVDRV::Close, "Close"}`.
    pub const CLOSE: u32 = 2;
    /// `{3, &NVDRV::Initialize, "Initialize"}`.
    pub const INITIALIZE: u32 = 3;
    /// `{4, &NVDRV::QueryEvent, "QueryEvent"}`.
    pub const QUERY_EVENT: u32 = 4;
    /// `{5, nullptr, "MapSharedMem"}`: not served.
    pub const MAP_SHARED_MEM: u32 = 5;
    /// `{6, &NVDRV::GetStatus, "GetStatus"}`.
    pub const GET_STATUS: u32 = 6;
    /// `{7, nullptr, "SetAruidForTest"}`: not served.
    pub const SET_ARUID_FOR_TEST: u32 = 7;
    /// `{8, &NVDRV::SetAruid, "SetAruid"}`.
    pub const SET_ARUID: u32 = 8;
    /// `{9, &NVDRV::DumpGraphicsMemoryInfo, "DumpGraphicsMemoryInfo"}`.
    pub const DUMP_GRAPHICS_MEMORY_INFO: u32 = 9;
    /// `{10, nullptr, "InitializeDevtools"}`: not served.
    pub const INITIALIZE_DEVTOOLS: u32 = 10;
    /// `{11, &NVDRV::Ioctl2, "Ioctl2"}`.
    pub const IOCTL2: u32 = 11;
    /// `{12, &NVDRV::Ioctl3, "Ioctl3"}`.
    pub const IOCTL3: u32 = 12;
    /// `{13, &NVDRV::SetGraphicsFirmwareMemoryMarginEnabled, ..}`.
    pub const SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED: u32 = 13;
}

use command::{
    CLOSE, DUMP_GRAPHICS_MEMORY_INFO, GET_STATUS, INITIALIZE, IOCTL, IOCTL2, IOCTL3, OPEN,
    QUERY_EVENT, SET_ARUID, SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED,
};

/// What an nvdrv command reports beside its IPC result (`nvdata.h`: `enum class NvResult`). Only
/// the codes nvsrv answers with are here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvResult {
    /// `Success = 0x0`.
    Success,
    /// `NotImplemented = 0x1`: an fd that is not open, or an ioctl the device does not know.
    NotImplemented,
    /// `NotSupported = 0x2`: the profiler device.
    NotSupported,
    /// `BadParameter = 0x4`: an event the device does not have.
    BadParameter,
    /// `InsufficientMemory = 0x6`.
    InsufficientMemory,
    /// `InvalidState = 0x8`: a negative fd.
    InvalidState,
    /// `InvalidSize = 0xA`: a submission with more entries than it carries.
    InvalidSize,
    /// `BadValue = 0xB`.
    BadValue,
    /// `AlreadyAllocated = 0xD`: a channel's second `AllocGPFIFOEx2`, a class slot taken.
    AlreadyAllocated,
    /// `Busy = 0xE`: a channel already bound to a timeslice group, and a `SubmitGPFIFO` past
    /// the cap on its channel's waiting submissions ([`gpu::submits`], "The cap").
    Busy,
    /// `FileOperationFailed = 0x30003`: a path no device has.
    FileOperationFailed,
    /// `AccessDenied = 0x30010`.
    AccessDenied,
}

impl NvResult {
    /// The word the reply carries.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => 0x0,
            Self::NotImplemented => 0x1,
            Self::NotSupported => 0x2,
            Self::BadParameter => 0x4,
            Self::InsufficientMemory => 0x6,
            Self::InvalidState => 0x8,
            Self::InvalidSize => 0xA,
            Self::BadValue => 0xB,
            Self::AlreadyAllocated => 0xD,
            Self::Busy => 0xE,
            Self::FileOperationFailed => 0x3_0003,
            Self::AccessDenied => 0x3_0010,
        }
    }
}

/// A device `Open` can name (Eden `Module::Module`'s `builders`, and PR's `/dev/nvhost-tsg-gpu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// `/dev/nvmap`: memory handles.
    Nvmap,
    /// `/dev/nvhost-ctrl-gpu`: the GPU's characteristics (slice 10).
    NvhostCtrlGpu,
    /// `/dev/nvhost-as-gpu`: address spaces (slice 10).
    NvhostAsGpu,
    /// `/dev/nvhost-tsg-gpu`: timeslice groups (slice 10; PR).
    NvhostTsgGpu,
    /// `/dev/nvhost-gpu`: channels and submission (slice 11).
    NvhostGpu,
    /// `/dev/nvhost-ctrl`: Switch 1's syncpoint control.
    NvhostCtrl,
    /// `/dev/nvdisp_disp0`: Switch 1's display.
    NvdispDisp0,
    /// `/dev/nvhost-nvdec`: Switch 1's video decoder.
    NvhostNvdec,
    /// `/dev/nvhost-nvjpg`: Switch 1's JPEG decoder.
    NvhostNvjpg,
    /// `/dev/nvhost-vic`: Switch 1's image compositor.
    NvhostVic,
}

/// Every device, in the order of [`Device`]'s variants.
pub const DEVICES: [Device; 10] = [
    Device::Nvmap,
    Device::NvhostCtrlGpu,
    Device::NvhostAsGpu,
    Device::NvhostTsgGpu,
    Device::NvhostGpu,
    Device::NvhostCtrl,
    Device::NvdispDisp0,
    Device::NvhostNvdec,
    Device::NvhostNvjpg,
    Device::NvhostVic,
];

/// The profiler's path: `Open` refuses it with `NotSupported` (`NVDRV::Open`: "cannot be opened
/// in production").
pub const PROFILER_PATH: &str = "/dev/nvhost-prof-gpu";

/// The longest path `Open` acts on: the longest of the devices' paths and [`PROFILER_PATH`].
pub const LONGEST_PATH: usize = longest_path();

const fn longest_path() -> usize {
    let mut longest = PROFILER_PATH.len();
    let mut index = 0;
    while index < DEVICES.len() {
        let len = DEVICES[index].path().len();
        if len > longest {
            longest = len;
        }
        index += 1;
    }
    longest
}

impl Device {
    /// The device's path.
    #[must_use]
    pub const fn path(self) -> &'static str {
        match self {
            Self::Nvmap => "/dev/nvmap",
            Self::NvhostCtrlGpu => "/dev/nvhost-ctrl-gpu",
            Self::NvhostAsGpu => "/dev/nvhost-as-gpu",
            Self::NvhostTsgGpu => "/dev/nvhost-tsg-gpu",
            Self::NvhostGpu => "/dev/nvhost-gpu",
            Self::NvhostCtrl => "/dev/nvhost-ctrl",
            Self::NvdispDisp0 => "/dev/nvdisp_disp0",
            Self::NvhostNvdec => "/dev/nvhost-nvdec",
            Self::NvhostNvjpg => "/dev/nvhost-nvjpg",
            Self::NvhostVic => "/dev/nvhost-vic",
        }
    }

    /// The device whose path is exactly `path`.
    #[must_use]
    pub fn from_path(path: &[u8]) -> Option<Self> {
        DEVICES
            .into_iter()
            .find(|device| device.path().as_bytes() == path)
    }
}

/// Which of the three ioctl commands carried an ioctl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoctlVariant {
    /// `Ioctl` (1): one in buffer, one out buffer.
    Ioctl,
    /// `Ioctl2` (11): and an inline in buffer.
    Ioctl2,
    /// `Ioctl3` (12): and an inline out buffer.
    Ioctl3,
}

impl IoctlVariant {
    /// The command's name, for logs.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Ioctl => "Ioctl",
            Self::Ioctl2 => "Ioctl2",
            Self::Ioctl3 => "Ioctl3",
        }
    }
}

/// What an ioctl answers: its result, the bytes for out buffer 0, and, for `Ioctl3`, the bytes
/// for out buffer 1 (the inline output), each written if the ioctl's `is_out` bit is set and
/// clipped to its buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoctlAnswer {
    /// The result word.
    pub result: NvResult,
    /// The argument the device gives back.
    pub output: Vec<u8>,
    /// What the device gives back inline (`Ioctl3`'s out buffer 1); empty for every other
    /// variant.
    pub inline: Vec<u8>,
}

impl IoctlAnswer {
    /// A result with nothing to give back.
    const fn only(result: NvResult) -> Self {
        Self::with(result, Vec::new())
    }

    /// A result and an argument, nothing inline.
    const fn with(result: NvResult, output: Vec<u8>) -> Self {
        Self {
            result,
            output,
            inline: Vec::new(),
        }
    }
}

/// One ioctl as a device sees it: the command that carried it, the fd, the number, the argument
/// read from in buffer 0, and the sizes of out buffer 0 and (for `Ioctl3`) out buffer 1, which
/// some devices check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoctlCall<'a> {
    /// `Ioctl`, `Ioctl2` or `Ioctl3`.
    pub variant: IoctlVariant,
    /// The fd.
    pub fd: i32,
    /// The ioctl number.
    pub ioctl: Ioctl,
    /// In buffer 0, up to [`MAX_ARGUMENT_BYTES`].
    pub input: &'a [u8],
    /// `Ioctl2`'s in buffer 1, the inline input, up to [`gpu::MAX_ENTRY_BYTES`]: read only for
    /// the ioctl that takes one ([`gpu::reads_inline_input`]), empty otherwise.
    pub inline_input: &'a [u8],
    /// Out buffer 0's size.
    pub out_len: usize,
    /// Out buffer 1's size for `Ioctl3`, 0 otherwise.
    pub inline_out_len: usize,
}

/// gsv's GPU memory, under gsv's shared lock, and the GPU's front: what nvhost-as-gpu acts on
/// ([`as_gpu`]); and, on a server with a channel front, what nvhost-gpu acts on ([`gpu`]).
pub struct GpuAccess<'a> {
    /// The bookkeeping gpu 33 and the gpu worker share.
    pub memory: &'a mut GpuMemory,
    /// The GPU's front, which a new address space is given to.
    pub front: &'a dyn GpuVmFront,
    /// The channels' wiring, if the server has a channel front.
    pub channels: Option<ChannelAccess<'a>>,
}

/// The fd `Open` answers when no device has the path (`nvdata.h`: `INVALID_NVDRV_FD = -1`).
pub const INVALID_FD: i32 = -1;

/// The first fd (`nvdrv.h`: `DeviceFD next_fd = 1`).
pub const FIRST_FD: u32 = 1;

/// One past the last fd: fds are `s32`s (`nvdata.h`: `using DeviceFD = s32`).
pub const FD_END: u32 = 1 << 31;

/// The most files open at once. A guess: Eden has no limit. It bounds what a title can make the
/// table hold.
pub const MAX_OPEN_FILES: usize = 0x1000;

/// What every nvsrv session shares (the nvdrv `Module`): the open files and the nvmap handles.
/// Plain data, saved in the server's snapshot. Deserialising goes through `SavedNvsrvState`,
/// which refuses a state the commands cannot reach ([`NvsrvStateError`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedNvsrvState")]
pub struct NvsrvState {
    files: BTreeMap<u32, Device>,
    next_fd: u32,
    nvmap: NvMap,
    as_gpu: AsGpu,
    tsg_gpu: TsgGpu,
    gpu: NvhostGpu,
    ctrl_gpu: CtrlGpu,
}

/// The serialised form of [`NvsrvState`], field for field, and what deserialising validates.
#[derive(Deserialize)]
struct SavedNvsrvState {
    files: BTreeMap<u32, Device>,
    next_fd: u32,
    nvmap: NvMap,
    as_gpu: AsGpu,
    tsg_gpu: TsgGpu,
    gpu: NvhostGpu,
    ctrl_gpu: CtrlGpu,
}

/// Why a saved [`NvsrvState`] could not be read back: its open files are not ones `Open` and
/// `Close` lead to. The checks run in this order, the first that fails reported; the nvmap
/// table is checked by its own type ([`nvmap::NvMapStateError`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NvsrvStateError {
    /// The next fd is below the first, or past the last `s32`.
    #[error("the next fd {next} is not one fds can reach")]
    NextFd {
        /// The next fd in the data.
        next: u32,
    },
    /// More files open than the table holds.
    #[error("{count} files are open, more than can be")]
    TooManyFiles {
        /// How many there are.
        count: usize,
    },
    /// An open fd below the first, or at or past the next, so never handed out.
    #[error("fd {fd} is open but was never handed out")]
    NotIssued {
        /// The fd.
        fd: u32,
    },
    /// An address-space view on an fd that is not an open `/dev/nvhost-as-gpu`.
    #[error("fd {fd} has an address-space view but is not an open nvhost-as-gpu")]
    NotAnAddressSpace {
        /// The fd.
        fd: u32,
    },
    /// A timeslice group on an fd that is not an open `/dev/nvhost-tsg-gpu`.
    #[error("fd {fd} has a timeslice group but is not an open nvhost-tsg-gpu")]
    NotATimesliceGroup {
        /// The fd.
        fd: u32,
    },
    /// A bound channel whose fd was never handed out.
    #[error("channel fd {fd} is bound to a timeslice group but was never handed out")]
    ChannelNotIssued {
        /// The channel's fd.
        fd: u32,
    },
    /// A bound channel whose fd is open as a device other than `/dev/nvhost-gpu`. A bind takes
    /// only an open nvhost-gpu fd, fds are never reused and a file's device never changes, so a
    /// bound channel's fd that is still open is an nvhost-gpu; one that was closed may stay
    /// bound.
    #[error("channel fd {fd} is bound to a timeslice group but is open as another device")]
    NotAChannel {
        /// The channel's fd.
        fd: u32,
    },
    /// An nvhost-gpu channel on an fd that is not an open `/dev/nvhost-gpu`: a channel goes
    /// when its fd closes.
    #[error("fd {fd} has a GPU channel but is not an open nvhost-gpu")]
    NotAGpuChannel {
        /// The fd.
        fd: u32,
    },
    /// An address-space binding on an fd that is not an open `/dev/nvhost-gpu`: `BindChannel`
    /// takes only an open nvhost-gpu fd, and the binding goes when the fd closes.
    #[error("fd {fd} is bound to an address space but is not an open nvhost-gpu")]
    NotABoundChannel {
        /// The fd.
        fd: u32,
    },
    /// A GPU channel made in an address space its fd is not bound to: `BindChannel` refuses
    /// another space once the channel is made, and the channel is made only in the bound one.
    #[error("channel fd {fd}'s GPU channel was made in an address space it is not bound to")]
    MadeUnbound {
        /// The channel's fd.
        fd: u32,
    },
    /// ctrl-gpu events on an fd that is not an open `/dev/nvhost-ctrl-gpu`: they are made when
    /// one opens and go when it closes.
    #[error("fd {fd} has ctrl-gpu events but is not an open nvhost-ctrl-gpu")]
    NotACtrlGpu {
        /// The fd.
        fd: u32,
    },
    /// An open `/dev/nvhost-ctrl-gpu` without its two events, which `Open` makes with it.
    #[error("nvhost-ctrl-gpu fd {fd} is open without its events")]
    CtrlGpuWithoutEvents {
        /// The fd.
        fd: u32,
    },
}

impl TryFrom<SavedNvsrvState> for NvsrvState {
    type Error = NvsrvStateError;

    fn try_from(saved: SavedNvsrvState) -> Result<Self, Self::Error> {
        let SavedNvsrvState {
            files,
            next_fd,
            nvmap,
            as_gpu,
            tsg_gpu,
            gpu,
            ctrl_gpu,
        } = saved;
        if !(FIRST_FD..=FD_END).contains(&next_fd) {
            return Err(NvsrvStateError::NextFd { next: next_fd });
        }
        if files.len() > MAX_OPEN_FILES {
            return Err(NvsrvStateError::TooManyFiles { count: files.len() });
        }
        if let Some(&fd) = files.keys().find(|&&fd| fd < FIRST_FD || fd >= next_fd) {
            return Err(NvsrvStateError::NotIssued { fd });
        }
        let open_as = |fd: &u32, device: Device| files.get(fd) == Some(&device);
        if let Some(fd) = as_gpu.fds().find(|fd| !open_as(fd, Device::NvhostAsGpu)) {
            return Err(NvsrvStateError::NotAnAddressSpace { fd });
        }
        if let Some(fd) = tsg_gpu.fds().find(|fd| !open_as(fd, Device::NvhostTsgGpu)) {
            return Err(NvsrvStateError::NotATimesliceGroup { fd });
        }
        if let Some(fd) = tsg_gpu
            .channels()
            .find(|&fd| fd < FIRST_FD || fd >= next_fd)
        {
            return Err(NvsrvStateError::ChannelNotIssued { fd });
        }
        if let Some(fd) = tsg_gpu.channels().find(|fd| {
            files
                .get(fd)
                .is_some_and(|&device| device != Device::NvhostGpu)
        }) {
            return Err(NvsrvStateError::NotAChannel { fd });
        }
        if let Some(fd) = gpu.fds().find(|fd| !open_as(fd, Device::NvhostGpu)) {
            return Err(NvsrvStateError::NotAGpuChannel { fd });
        }
        if let Some((fd, _)) = as_gpu
            .bindings()
            .find(|(fd, _)| !open_as(fd, Device::NvhostGpu))
        {
            return Err(NvsrvStateError::NotABoundChannel { fd });
        }
        if let Some(fd) = gpu.fds().find(|&fd| {
            gpu.made_in(fd)
                .is_some_and(|space| as_gpu.bound(fd) != Some(space))
        }) {
            return Err(NvsrvStateError::MadeUnbound { fd });
        }
        if let Some(fd) = ctrl_gpu
            .fds()
            .find(|fd| !open_as(fd, Device::NvhostCtrlGpu))
        {
            return Err(NvsrvStateError::NotACtrlGpu { fd });
        }
        if let Some((&fd, _)) = files.iter().find(|&(&fd, &device)| {
            device == Device::NvhostCtrlGpu && ctrl_gpu.events(fd).is_none()
        }) {
            return Err(NvsrvStateError::CtrlGpuWithoutEvents { fd });
        }
        Ok(Self {
            files,
            next_fd,
            nvmap,
            as_gpu,
            tsg_gpu,
            gpu,
            ctrl_gpu,
        })
    }
}

/// `path` up to its first zero byte (`Common::StringFromBuffer`).
fn up_to_zero(path: &[u8]) -> &[u8] {
    path.split(|&byte| byte == 0).next().unwrap_or_default()
}

/// Why `Open` refuses a path (module docs, "Devices and fds").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// The profiler's path.
    Profiler,
    /// A path no device has.
    NoDevice,
    /// [`MAX_OPEN_FILES`] files open.
    TooManyFiles(Device),
    /// The next fd is past the last `s32`.
    FdsUsedUp(Device),
}

impl Refusal {
    /// The fd and result `Open` of `path` answers, logged.
    fn answer(self, path: &[u8]) -> (i32, NvResult) {
        match self {
            Self::Profiler => {
                tracing::warn!("nvsrv: the profiler cannot be opened in production");
                (0, NvResult::NotSupported)
            }
            Self::NoDevice => {
                tracing::warn!(
                    path = %String::from_utf8_lossy(up_to_zero(path)),
                    "nvsrv: open of a device that does not exist"
                );
                (INVALID_FD, NvResult::FileOperationFailed)
            }
            Self::TooManyFiles(device) => {
                tracing::warn!(device = device.path(), "nvsrv: too many files open");
                (INVALID_FD, NvResult::FileOperationFailed)
            }
            Self::FdsUsedUp(device) => {
                tracing::warn!(device = device.path(), "nvsrv: fds are used up");
                (INVALID_FD, NvResult::FileOperationFailed)
            }
        }
    }
}

/// The open-file table's key for `fd`, or `InvalidState` for a negative fd (`Module::VerifyFD`:
/// `if (fd < 0)`).
fn key(fd: i32) -> Result<u32, NvResult> {
    u32::try_from(fd).map_err(|_| NvResult::InvalidState)
}

impl Default for NvsrvState {
    fn default() -> Self {
        Self::new()
    }
}

impl NvsrvState {
    /// Nothing open, no handles; the first fd is [`FIRST_FD`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: BTreeMap::new(),
            next_fd: FIRST_FD,
            nvmap: NvMap::new(),
            as_gpu: AsGpu::new(),
            tsg_gpu: TsgGpu::new(),
            gpu: NvhostGpu::new(),
            ctrl_gpu: CtrlGpu::new(),
        }
    }

    /// The nvhost-gpu fds' channels.
    #[must_use]
    pub const fn gpu(&self) -> &NvhostGpu {
        &self.gpu
    }

    /// The nvmap handle table.
    #[must_use]
    pub const fn nvmap(&self) -> &NvMap {
        &self.nvmap
    }

    /// The nvhost-as-gpu fds' views.
    #[must_use]
    pub const fn as_gpu(&self) -> &AsGpu {
        &self.as_gpu
    }

    /// The nvhost-tsg-gpu fds' groups.
    #[must_use]
    pub const fn tsg_gpu(&self) -> &TsgGpu {
        &self.tsg_gpu
    }

    /// The nvhost-ctrl-gpu fds' events.
    #[must_use]
    pub const fn ctrl_gpu(&self) -> &CtrlGpu {
        &self.ctrl_gpu
    }

    /// How many files are open.
    #[must_use]
    pub fn open_files(&self) -> usize {
        self.files.len()
    }

    /// The device open as `fd`, or what a command naming `fd` answers (`Module::VerifyFD`):
    /// `InvalidState` for a negative fd, `NotImplemented` for one not open.
    pub fn file(&self, fd: i32) -> Result<Device, NvResult> {
        let key = key(fd)?;
        self.files
            .get(&key)
            .copied()
            .ok_or(NvResult::NotImplemented)
    }

    /// The device `Open` of `path` would open and the fd it would answer, or why it would
    /// refuse (module docs, "Devices and fds"). Nothing is changed or logged.
    fn admit(&self, path: &[u8]) -> Result<(Device, i32), Refusal> {
        let path = up_to_zero(path);
        if path == PROFILER_PATH.as_bytes() {
            return Err(Refusal::Profiler);
        }
        let device = Device::from_path(path).ok_or(Refusal::NoDevice)?;
        if self.files.len() >= MAX_OPEN_FILES {
            return Err(Refusal::TooManyFiles(device));
        }
        let fd = i32::try_from(self.next_fd).map_err(|_| Refusal::FdsUsedUp(device))?;
        Ok((device, fd))
    }

    /// Whether `Open` of `path` would open an nvhost-ctrl-gpu, whose two events the caller makes
    /// first and gives [`open`](Self::open) (module docs, "Devices and fds").
    #[must_use]
    pub fn opens_ctrl_gpu(&self, path: &[u8]) -> bool {
        matches!(self.admit(path), Ok((Device::NvhostCtrlGpu, _)))
    }

    /// `Open`: the device whose path `path` holds, up to its first zero byte (module docs,
    /// "Devices and fds"). The fd, and the result. `events` are an nvhost-ctrl-gpu's two
    /// events, made by the caller beforehand ([`opens_ctrl_gpu`](Self::opens_ctrl_gpu)): an
    /// nvhost-ctrl-gpu without them, or with a pair [`CtrlGpu::check`] refuses, is not opened (fd
    /// -1 and `FileOperationFailed`, logged as an error: the caller's fault); given for another
    /// device they are not kept, which is logged as an error and changes nothing else. The logs
    /// are log-only: no test captures them.
    pub fn open(&mut self, path: &[u8], events: Option<CtrlGpuEvents>) -> (i32, NvResult) {
        let (device, fd) = match self.admit(path) {
            Ok(admitted) => admitted,
            Err(refusal) => return refusal.answer(path),
        };
        let key = self.next_fd;
        match (device, events) {
            (Device::NvhostCtrlGpu, None) => {
                tracing::error!("nvsrv: nvhost-ctrl-gpu opened without its events");
                return (INVALID_FD, NvResult::FileOperationFailed);
            }
            (Device::NvhostCtrlGpu, Some(events)) => {
                if let Err(error) = self.ctrl_gpu.keep(key, events) {
                    tracing::error!(%error, "nvsrv: nvhost-ctrl-gpu's events refused");
                    return (INVALID_FD, NvResult::FileOperationFailed);
                }
            }
            (_, Some(_)) => {
                tracing::error!(
                    device = device.path(),
                    "nvsrv: events for a device without them"
                );
            }
            (_, None) => {}
        }
        self.files.insert(key, device);
        self.next_fd += 1;
        (fd, NvResult::Success)
    }

    /// `Close` (`Module::Close`): an as-gpu fd's view, a tsg-gpu fd's group and an nvhost-gpu
    /// fd's channel go with it (its waiting submissions answered, its GPU channel destroyed and
    /// its syncpoint freed, [`gpu`]), and then the channel's binding; an address space whose last
    /// holder that was is destroyed in `gpu` ([`as_gpu`], "Closing").
    pub fn close(&mut self, fd: i32, mut gpu: GpuAccess<'_>) -> NvResult {
        let key = match key(fd) {
            Ok(key) => key,
            Err(result) => return result,
        };
        if self.files.remove(&key).is_none() {
            return NvResult::NotImplemented;
        }
        self.as_gpu.close(key, &mut gpu);
        self.tsg_gpu.close(key);
        self.ctrl_gpu.close(key);
        self.gpu.close(key, gpu.channels.as_mut());
        self.as_gpu.unbind(key, &mut gpu);
        NvResult::Success
    }

    /// `QueryEvent` (`Module::QueryEvent`): the fd checks, then the event (module docs,
    /// "Devices and fds"). Only nvhost-gpu and nvhost-ctrl-gpu have events.
    pub fn query_event(&self, fd: i32, event: u32) -> Result<Queried, NvResult> {
        let device = self.file(fd)?;
        let queried = match device {
            Device::NvhostGpu => self.gpu.event(key(fd)?, event),
            Device::NvhostCtrlGpu => self.ctrl_gpu.event(key(fd)?, event),
            _ => Queried::Missing,
        };
        if queried == Queried::Missing {
            tracing::warn!(
                device = device.path(),
                event,
                "nvsrv: an event no device has"
            );
        }
        Ok(queried)
    }

    /// Keeps `made` as nvhost-gpu fd `fd`'s event `event`, made for a [`Queried::Make`].
    pub fn set_event(&mut self, fd: i32, event: u32, made: starquake_kernel::id::ObjectId) {
        if let Ok(key) = key(fd) {
            self.gpu.set_event(key, event, made);
        }
    }

    /// `call`: the fd checks, then the device, then the policy for an ioctl the device does not
    /// know (module docs, "Ioctls"). nvhost-as-gpu and nvhost-gpu act on `gpu`; a submission that
    /// must wait for room is left in `gpu`'s channel access ([`ChannelAccess::deferred`]).
    pub fn ioctl(
        &mut self,
        policy: UnknownPolicy,
        call: &IoctlCall<'_>,
        mut gpu: GpuAccess<'_>,
    ) -> IoctlAnswer {
        let IoctlCall {
            variant, fd, ioctl, ..
        } = *call;
        let device = match self.file(fd) {
            Ok(device) => device,
            Err(result) => return IoctlAnswer::only(result),
        };
        // `file` checked the fd, so it is not negative.
        let key = key(fd).unwrap_or_default();
        let known = match (device, variant) {
            (Device::Nvmap, IoctlVariant::Ioctl) => self
                .nvmap
                .ioctl(ioctl, call.input)
                .map(|(result, output)| IoctlAnswer::with(result, output)),
            (Device::NvhostCtrlGpu, IoctlVariant::Ioctl) => ctrl_gpu::ioctl1(call),
            (Device::NvhostCtrlGpu, IoctlVariant::Ioctl3) => ctrl_gpu::ioctl3(call),
            (Device::NvhostAsGpu, _)
                if gpu.channels.is_some() && as_gpu::is_bind_channel(variant, ioctl) =>
            {
                let channels = &self.gpu;
                let made_in = |fd: u32| channels.made_in(fd);
                let files = &self.files;
                Some(
                    self.as_gpu
                        .bind_channel(key, call, files, &made_in, &mut gpu),
                )
            }
            (Device::NvhostAsGpu, _) => self.as_gpu.ioctl(key, call, gpu),
            (Device::NvhostTsgGpu, _) => self.tsg_gpu.ioctl(key, call, &self.files),
            (Device::NvhostGpu, _) => match gpu.channels.as_mut() {
                Some(access) => self.gpu.ioctl(key, call, &self.as_gpu, access),
                None => None,
            },
            _ => None,
        };
        if let Some(answer) = known {
            return answer;
        }
        tracing::warn!(
            device = device.path(),
            variant = variant.word(),
            ioctl = ioctl.raw(),
            group = ioctl.group(),
            command = ioctl.command(),
            "nvsrv: an ioctl the device does not know"
        );
        match policy {
            UnknownPolicy::Strict => IoctlAnswer::only(NvResult::NotImplemented),
            UnknownPolicy::AutoStub => IoctlAnswer::with(
                NvResult::Success,
                vec![0; usize::try_from(ioctl.size()).unwrap_or_default()],
            ),
        }
    }
}

/// The nvdrv interface: what a session on the nvsrv server is. It keeps nothing; the state is
/// the server's ([`NvsrvState`]).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Nvsrv;

commands! {
    impl Nvsrv {
        OPEN => open [In(MapAlias)],
        IOCTL => ioctl [In(AutoSelect), Out(AutoSelect)],
        CLOSE => close,
        INITIALIZE => initialize,
        QUERY_EVENT => query_event,
        GET_STATUS => get_status,
        SET_ARUID => set_aruid,
        DUMP_GRAPHICS_MEMORY_INFO => result_only,
        IOCTL2 => ioctl2 [In(AutoSelect), In(AutoSelect), Out(AutoSelect)],
        IOCTL3 => ioctl3 [In(AutoSelect), Out(AutoSelect), Out(AutoSelect)],
        SET_GRAPHICS_FIRMWARE_MEMORY_MARGIN_ENABLED => result_only,
    }
}

/// Reads up to `limit` bytes from the start of in buffer `index`.
fn read_in(
    cx: &mut Context<'_>,
    req: &Request<'_>,
    index: usize,
    limit: usize,
) -> Result<Vec<u8>, Fault> {
    let mut buffer = req.in_buffer(cx, index)?;
    let len = usize::try_from(buffer.len()).map_or(limit, |len| len.min(limit));
    let mut bytes = vec![0; len];
    buffer.read(&mut bytes)?;
    Ok(bytes)
}

impl Nvsrv {
    /// `Open` (0). in buffer 0: the path. out: `s32` fd, the result.
    fn open(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        // One byte past the longest path: a path that long has no zero byte where any device's
        // would end, so it names none, and nothing further need be read.
        let path = read_in(cx, req, 0, LONGEST_PATH + 1)?;
        // An nvhost-ctrl-gpu's events are made before it opens (module docs, "Devices and fds").
        let events = if cx.family::<NvsrvFamily>()?.state.opens_ctrl_gpu(&path) {
            Some(CtrlGpuEvents {
                error_notifier: cx.create_event()?,
                unknown: cx.create_event()?,
            })
        } else {
            None
        };
        let (fd, result) = cx.family::<NvsrvFamily>()?.state.open(&path, events);
        reply.payload().write_i32(fd).write_u32(result.raw());
        reply.ok()
    }

    /// `Ioctl` (1).
    fn ioctl(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        run_ioctl(IoctlVariant::Ioctl, cx, req, reply)
    }

    /// `Ioctl2` (11): its inline input (in buffer 1) is read only for nvhost-gpu's `SubmitGPFIFO`
    /// 0x1B (module docs).
    fn ioctl2(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        run_ioctl(IoctlVariant::Ioctl2, cx, req, reply)
    }

    /// `Ioctl3` (12): its inline output (out buffer 1) is not written (module docs).
    fn ioctl3(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        run_ioctl(IoctlVariant::Ioctl3, cx, req, reply)
    }

    /// `Close` (2). in: `s32` fd. out: the result.
    fn close(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let fd = req.payload().read_i32()?;
        let result = cx.family::<NvsrvFamily>()?.close(fd);
        reply.payload().write_u32(result.raw());
        reply.ok()
    }

    /// `Initialize` (3): the session is initialised already, so nothing is read.
    fn initialize(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(NvResult::Success.raw());
        reply.ok()
    }

    /// `QueryEvent` (4). in: `s32` fd, `u32` event id. out: the event, copied, and `Success`;
    /// or the result alone (module docs, "Devices and fds").
    fn query_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let fd = payload.read_i32()?;
        let event = payload.read_u32()?;
        let queried = cx.family::<NvsrvFamily>()?.state.query_event(fd, event);
        let made = match queried {
            Err(result) => {
                reply.payload().write_u32(result.raw());
                return reply.ok();
            }
            Ok(Queried::Missing) => {
                reply.payload().write_u32(NvResult::BadParameter.raw());
                return reply.ok();
            }
            Ok(Queried::Event(made)) => made,
            Ok(Queried::Make) => {
                let made = cx.create_event()?;
                cx.family::<NvsrvFamily>()?.state.set_event(fd, event, made);
                made
            }
        };
        reply.copy_handle(made);
        reply.payload().write_u32(NvResult::Success.raw());
        reply.ok()
    }

    /// `GetStatus` (6): Eden's stub, `Success`.
    fn get_status(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        tracing::info!("nvsrv: GetStatus stub");
        reply.payload().write_u32(NvResult::Success.raw());
        reply.ok()
    }

    /// `SetAruid` (8). in: `u64` applet resource user id. out: `Success`.
    fn set_aruid(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let aruid = req.payload().read_u64()?;
        tracing::info!(aruid, "nvsrv: SetAruid stub");
        reply.payload().write_u32(NvResult::Success.raw());
        reply.ok()
    }

    /// `DumpGraphicsMemoryInfo` (9) and `SetGraphicsFirmwareMemoryMarginEnabled` (13): Eden's
    /// stubs, a result and nothing else.
    fn result_only(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        tracing::info!(command = req.command_id(), "nvsrv: stub");
        reply.ok()
    }
}

/// `Ioctl`, `Ioctl2` and `Ioctl3`. in: `s32` fd, `u32` ioctl; in buffer 0, the argument. out:
/// out buffer 0, the argument back if the ioctl's `is_out` bit is set; the result. A submission
/// that must wait for room ([`gpu`], "The ioctls") writes its out buffer at once and defers the
/// result; if that write or the reply token fails, the request answers the fault and the
/// submission waits all the same, with no token ([`gpu`], "A full queue").
fn run_ioctl(
    variant: IoctlVariant,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let mut payload = req.payload();
    let fd = payload.read_i32()?;
    let ioctl = Ioctl::from_raw(payload.read_u32()?);
    let limit = usize::try_from(MAX_ARGUMENT_BYTES).unwrap_or_default();
    let input = read_in(cx, req, 0, limit)?;
    let inline_input = if gpu::reads_inline_input(variant, ioctl)
        && cx.family::<NvsrvFamily>()?.reads_channels(fd)
    {
        read_in(cx, req, 1, gpu::MAX_ENTRY_BYTES)?
    } else {
        Vec::new()
    };
    let out_len = out_buffer_len(cx, req, 0)?;
    let inline_out_len = if variant == IoctlVariant::Ioctl3 {
        out_buffer_len(cx, req, 1)?
    } else {
        0
    };
    let call = IoctlCall {
        variant,
        fd,
        ioctl,
        input: &input,
        inline_input: &inline_input,
        out_len,
        inline_out_len,
    };
    let policy = cx.policy();
    let (answer, deferral) = cx.family::<NvsrvFamily>()?.ioctl(policy, &call);
    settle_device_shares(cx)?;
    let written = if ioctl.is_out() {
        write_out(cx, req, 0, &answer.output).and_then(|()| {
            if variant == IoctlVariant::Ioctl3 && !answer.inline.is_empty() {
                write_out(cx, req, 1, &answer.inline)
            } else {
                Ok(())
            }
        })
    } else {
        Ok(())
    };
    let Some(deferral) = deferral else {
        written?;
        reply.payload().write_u32(answer.result.raw());
        return reply.ok();
    };
    // The channel's syncpoint maximum is raised already, so the submission waits whatever
    // happens to the reply: without a token if the out buffer or the token failed ([`gpu`],
    // "A full queue").
    match written.and_then(|()| cx.defer()) {
        Ok(token) => {
            cx.family::<NvsrvFamily>()?
                .wait_for_room(deferral, Some(token));
            reply.deferred()
        }
        Err(fault) => {
            cx.family::<NvsrvFamily>()?.wait_for_room(deferral, None);
            Err(fault)
        }
    }
}

/// Takes the device-share steps nvmap's ioctl left through the kernel ([`nvmap`], "The
/// device-share lock"), oldest first: a lock taken is kept by its handle, and a refusal of a lock
/// or a release is logged as an error, as Eden's `ASSERT` logs it, and changes no answer. A lock
/// its handle cannot keep (only another caller of the table, between the ioctl and this, could
/// change it) is released at once. Fails only when the server has no nvsrv family, as the ioctl
/// that left the steps would have.
fn settle_device_shares(cx: &mut Context<'_>) -> Result<(), Fault> {
    let steps = cx.family::<NvsrvFamily>()?.state.nvmap.take_steps();
    for step in steps {
        let token = match step {
            DeviceShareStep::Lock {
                handle,
                address,
                size,
            } => match cx.lock_device_share(address, size) {
                Ok(token) => match cx.family::<NvsrvFamily>()?.state.nvmap.keep_lock(handle, token)
                {
                    Ok(()) => continue,
                    Err(token) => {
                        tracing::error!(handle, "nvmap: a lock its handle cannot keep");
                        token
                    }
                },
                Err(error) => {
                    tracing::error!(
                        %error,
                        handle,
                        address,
                        size,
                        "nvmap: the kernel refused an allocation's device-share lock"
                    );
                    continue;
                }
            },
            DeviceShareStep::Unlock(token) => token,
        };
        if let Err(error) = cx.unlock_device_share(token) {
            tracing::error!(%error, ?token, "nvmap: the kernel refused a device-share release");
        }
    }
    Ok(())
}

/// Out buffer `index`'s size (Eden `GetWriteBufferSize`). The framework refuses a request that
/// lacks a declared buffer before the handler runs, so the buffer is there.
fn out_buffer_len(cx: &mut Context<'_>, req: &Request<'_>, index: usize) -> Result<usize, Fault> {
    let len = req.out_buffer(cx, index)?.len();
    Ok(usize::try_from(len).unwrap_or(usize::MAX))
}

/// Writes `bytes` to the start of out buffer `index`, clipped to the buffer.
fn write_out(
    cx: &mut Context<'_>,
    req: &Request<'_>,
    index: usize,
    bytes: &[u8],
) -> Result<(), Fault> {
    let mut out = req.out_buffer(cx, index)?;
    let len = usize::try_from(out.len()).map_or(bytes.len(), |len| len.min(bytes.len()));
    out.write(bytes.get(..len).unwrap_or_default())
}

/// The interfaces an nvsrv server's sessions can be.
#[must_use]
pub fn nvsrv_interfaces() -> Interfaces {
    Interfaces::new().with::<Nvsrv>()
}

/// The nvsrv server's family: the state every session shares ([`NvsrvState`]), and the wiring
/// nvhost-as-gpu and nvhost-gpu reach gsv through: gsv's shared state, the GPU's front and, once
/// given, its channel front. Only the state is saved; the wiring is the run's.
pub struct NvsrvFamily {
    state: NvsrvState,
    shared: Arc<Shared>,
    front: Arc<dyn GpuVmFront>,
    channels: Option<Arc<dyn ChannelFront>>,
}

impl NvsrvFamily {
    /// A new state over `shared`'s GPU memory and `front`, with no channel front: nvhost-gpu's
    /// ioctls go to the unknown-ioctl policy until [`with_channel_front`](Self::with_channel_front).
    #[must_use]
    pub fn new(shared: Arc<Shared>, front: Arc<dyn GpuVmFront>) -> Self {
        Self {
            state: NvsrvState::new(),
            shared,
            front,
            channels: None,
        }
    }

    /// The same wiring with `channels` as the channel front nvhost-gpu's channels are made on
    /// ([`gpu`]).
    #[must_use]
    pub fn with_channel_front(self, channels: Arc<dyn ChannelFront>) -> Self {
        Self {
            channels: Some(channels),
            ..self
        }
    }

    /// Whether an ioctl on `fd` reaches a channel: the server has a channel front and `fd` is
    /// an open nvhost-gpu.
    fn reads_channels(&self, fd: i32) -> bool {
        self.channels.is_some() && self.state.file(fd) == Ok(Device::NvhostGpu)
    }

    /// The same wiring with `state` in place of the state.
    #[must_use]
    pub fn with_state(self, state: NvsrvState) -> Self {
        Self { state, ..self }
    }

    /// The state.
    #[must_use]
    pub const fn state(&self) -> &NvsrvState {
        &self.state
    }

    /// [`NvsrvState::ioctl`] with gsv's shared lock held, taken for every ioctl. No device calls
    /// the kernel, so nothing is called under it but the GPU's fronts, which never call back into
    /// gsv (`crate::gpu`, "The server"). The answer, and the submission that must wait for room,
    /// if one must ([`wait_for_room`](Self::wait_for_room)).
    pub fn ioctl(
        &mut self,
        policy: UnknownPolicy,
        call: &IoctlCall<'_>,
    ) -> (IoctlAnswer, Option<Deferral>) {
        let shared = Arc::clone(&self.shared);
        let syncpoints = Arc::clone(shared.syncpoints());
        let mut locked = shared.lock();
        let (memory, submits) = locked.gpu_and_submits();
        let mut deferred = None;
        let channels = self.channels.as_ref().map(|front| ChannelAccess {
            submits,
            syncpoints: &syncpoints,
            front,
            deferred: &mut deferred,
        });
        let gpu = GpuAccess {
            memory,
            front: &*self.front,
            channels,
        };
        let answer = self.state.ioctl(policy, call, gpu);
        (answer, deferred)
    }

    /// `deferral`'s submission waits for room in gsv's shared state, `token` answering its
    /// request if it has one (none when its out buffer or its token failed: the submission's
    /// fence was given out, so it waits all the same); the channel's queue is tried once more
    /// under gsv's lock, so a room report that came since the front answered `Full` is not lost
    /// ([`gpu::submits`]).
    pub fn wait_for_room(&mut self, deferral: Deferral, token: Option<ReplyToken>) {
        let Some(front) = self.channels.as_ref() else {
            // Only a server with a channel front defers a submission.
            tracing::error!("nvsrv: a submission waits with no channel front");
            if let Some(token) = token {
                token.fail(NOT_SUPPORTED);
            }
            return;
        };
        let front = Arc::clone(front);
        let mut locked = self.shared.lock();
        let Deferral {
            channel,
            submission,
        } = deferral;
        // The ioctl refused a submission past the cap before raising its fence, and holds the
        // family between that check and this push, so this refusal is only for another caller
        // of `Submits::push`; the push has answered the token `Busy` (`gpu::submits`, "The cap").
        if let Err(error) = locked.submits().push(channel, token, submission, front) {
            tracing::error!(%error, "nvsrv: a waiting submission passed the cap");
        }
    }

    /// [`NvsrvState::close`] with gsv's shared lock held, as for [`ioctl`](Self::ioctl): the
    /// close may destroy an address space in gsv's GPU memory, a GPU channel, and answer the
    /// channel's waiting submissions.
    pub fn close(&mut self, fd: i32) -> NvResult {
        let shared = Arc::clone(&self.shared);
        let syncpoints = Arc::clone(shared.syncpoints());
        let mut locked = shared.lock();
        let (memory, submits) = locked.gpu_and_submits();
        let mut deferred = None;
        let channels = self.channels.as_ref().map(|front| ChannelAccess {
            submits,
            syncpoints: &syncpoints,
            front,
            deferred: &mut deferred,
        });
        let gpu = GpuAccess {
            memory,
            front: &*self.front,
            channels,
        };
        self.state.close(fd, gpu)
    }
}

impl Family for NvsrvFamily {
    /// The state, as [`NvsrvState`] saves itself.
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        self.state.save_state()
    }

    /// Replaces the state with what `bytes` hold, validated ([`NvsrvStateError`]); on an error
    /// the state is unchanged. The wiring stays.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.state.load_state(bytes)
    }

    /// When the server's last session has closed (`open` is 0), Eden's `Container::CloseSession`
    /// for the process: its nvmap handles are freed ([`NvMap::close_session`]), and the lock of
    /// each that went is posted to gsv's completion queue as a release, which the timer
    /// thread's drain takes to the kernel ([`nvmap`], "When the process's last session closes").
    /// Called with the kernel's lock held when a process's end closed the session: it touches
    /// only the family's own state and the queue, whose push takes only the queue's lock
    /// ([`crate::completion`], "Never blocking"), never gsv's shared lock or the kernel.
    fn session_closed(&mut self, open: usize) {
        if open != 0 {
            return;
        }
        for token in self.state.nvmap.close_session() {
            self.shared
                .queue()
                .push(Completion::DeviceShareReleased { token });
        }
    }
}

/// The nvsrv server: every session an [`Nvsrv`], sharing one [`NvsrvState`], whose
/// nvhost-as-gpu fds act on `shared`'s GPU memory and `front`. `policy` answers unknown commands
/// and, inside `Ioctl`, unknown ioctls (module docs).
#[must_use]
pub fn nvsrv_server(
    shared: Arc<Shared>,
    front: Arc<dyn GpuVmFront>,
    policy: UnknownPolicy,
) -> Server {
    Server::new(NVSRV_NAME, policy)
        .with_root(|| Box::new(Nvsrv))
        .with_interfaces(nvsrv_interfaces())
        .with_family(NvsrvFamily::new(shared, front))
}
