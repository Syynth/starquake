//! nvsrv: the NVIDIA driver session gsv 1402 returns, which speaks the old nvdrv
//! `INvDrvServices` protocol (`Open`, the three `Ioctl` variants, `Close`, `QueryEvent`) over the devices
//! Switch 2's NvRm opens (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv";
//! port slice 9).
//!
//! * [`ioctl`]: ioctl numbers, and the fixed-size argument structs devices read.
//! * [`nvmap`]: `/dev/nvmap` and its handle table.
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
//! and Switch 2's NvRm opens (rows 26, 45). The path is the buffer's bytes up to the first zero
//! byte (`Common::StringFromBuffer`). It answers:
//!
//! - `/dev/nvhost-prof-gpu` ([`PROFILER_PATH`]): fd 0 and `NotSupported` ("cannot be opened in
//!   production"), before anything else is looked at;
//! - any other path no device has: fd -1 ([`INVALID_FD`]) and `FileOperationFailed`;
//! - otherwise the next fd, counting up from 1 and never reused (`Module::next_fd`), and
//!   `Success`.
//!
//! The open files are the family's ([`NvsrvState`]): in both references every nvsrv session
//! shares the one nvdrv `Module`, its fds and its nvmap handles. Closing an IPC session closes
//! none of its fds, as in Eden, where they belong to the module.
//!
//! An fd a command names is checked as Eden's `Module` checks it: negative → `InvalidState`, not
//! open → `NotImplemented`, before the device is asked anything.
//!
//! **What each device does in this slice.** `/dev/nvmap` has its ioctls ([`nvmap`]). The others
//! open, but have no ioctls yet: ctrl-gpu, as-gpu and tsg-gpu come with port slice 10,
//! nvhost-gpu with slice 11, and the rest are Switch 1 devices (decision 11 builds only what
//! Switch 2's driver uses). No device has events yet, so `QueryEvent` on an open fd answers
//! `BadParameter`, Eden's answer for a device with none (`nvdevice::QueryEvent` returns
//! `nullptr`); ctrl-gpu's and nvhost-gpu's events come with their slices.
//!
//! # Ioctls
//!
//! `Ioctl`, `Ioctl2` and `Ioctl3` read the fd and the ioctl number, read the in buffer, check
//! the fd, and hand the device the ioctl ([`ioctl::Ioctl`]) and the argument. If the ioctl's
//! [`is_out`](ioctl::Ioctl::is_out) bit is set, what the device gives back is written to out
//! buffer 0, clipped to the buffer (`WrapGeneric`: `std::min(output.size(), sizeof(FixedArg))`),
//! whatever the result.
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
//! its payload; nvsrv has no input from the host, so it writes no notes. The open files and the
//! nvmap handles are the family state, plain data saved with the server, and restoring checks
//! every relation between their fields ([`NvsrvStateError`], [`nvmap::NvMapStateError`]).
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `nvsrv/tests.rs`:
//!
//! | Case | Eden (and NX2, PR) | Here |
//! |---|---|---|
//! | An in buffer larger than [`ioctl::MAX_ARGUMENT_BYTES`] | read whole | its first [`ioctl::MAX_ARGUMENT_BYTES`] bytes are read, the rest never touched: no device of this slice reads past its fixed struct, and a title cannot make the server copy more (`an_in_buffer_is_read_up_to_the_largest_argument`) |
//! | An out buffer, written when `is_out` is set | written whole, past what the device gave back too, with whatever the server's scratch buffer held; also on an fd error | written only as far as the device's answer (or the auto-stub's zeros); nothing on an fd error or a strict unknown ioctl (`an_out_buffer_gets_only_the_answer`, `fd_checks_come_before_the_policy`) |
//! | `Ioctl2`'s inline input, `Ioctl3`'s inline output | read; written when `is_out` is set | not read, not written: no device of this slice takes or gives one (`inline_buffers_are_left_alone`) |
//! | A path no device has, under auto-stub | NX2 opens any `/dev/nvhost-*` path as a stub device | refused under both policies, as Eden refuses it (`an_unknown_path_is_refused_under_both_policies`) |
//! | [`MAX_OPEN_FILES`] files open, or fds used up | no limit; the next fd overflows its `s32` | fd -1 and `FileOperationFailed` (a guess), nothing opened and no fd used (`open_refuses_a_full_table_and_spent_fds`) |
//! | A TIPC request | NX2 and PR: auto-stubbed or refused, as for any command they lack | the server's unknown-command policy (`a_tipc_request_goes_to_the_policy`) |
//!
//! A change an ioctl made stands if writing its out buffer then fails (the caller's memory is not
//! mapped): the caller gets the kernel's error, as for any service whose reply writes a buffer.
//! In Eden the write is dropped and the change stands too.

pub mod ioctl;
pub mod nvmap;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_services::commands;
use starquake_services::framework::{
    Context, Fault, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy,
};

use self::ioctl::{Ioctl, MAX_ARGUMENT_BYTES};
use self::nvmap::NvMap;

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
    /// `BadValue = 0xB`.
    BadValue,
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
            Self::BadValue => 0xB,
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

/// What an ioctl answers: its result, and the bytes for out buffer 0 if the ioctl's `is_out` bit
/// is set (clipped to the buffer when written).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoctlAnswer {
    /// The result word.
    pub result: NvResult,
    /// The argument the device gives back.
    pub output: Vec<u8>,
}

impl IoctlAnswer {
    /// A result with nothing to give back.
    const fn only(result: NvResult) -> Self {
        Self {
            result,
            output: Vec::new(),
        }
    }
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
}

/// The serialised form of [`NvsrvState`], field for field, and what deserialising validates.
#[derive(Deserialize)]
struct SavedNvsrvState {
    files: BTreeMap<u32, Device>,
    next_fd: u32,
    nvmap: NvMap,
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
}

impl TryFrom<SavedNvsrvState> for NvsrvState {
    type Error = NvsrvStateError;

    fn try_from(saved: SavedNvsrvState) -> Result<Self, Self::Error> {
        let SavedNvsrvState {
            files,
            next_fd,
            nvmap,
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
        Ok(Self {
            files,
            next_fd,
            nvmap,
        })
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
        }
    }

    /// The nvmap handle table.
    #[must_use]
    pub const fn nvmap(&self) -> &NvMap {
        &self.nvmap
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

    /// `Open`: the device whose path `path` holds, up to its first zero byte (module docs,
    /// "Devices and fds"). The fd, and the result.
    pub fn open(&mut self, path: &[u8]) -> (i32, NvResult) {
        let path = path.split(|&byte| byte == 0).next().unwrap_or_default();
        if path == PROFILER_PATH.as_bytes() {
            tracing::warn!("nvsrv: the profiler cannot be opened in production");
            return (0, NvResult::NotSupported);
        }
        let Some(device) = Device::from_path(path) else {
            tracing::warn!(
                path = %String::from_utf8_lossy(path),
                "nvsrv: open of a device that does not exist"
            );
            return (INVALID_FD, NvResult::FileOperationFailed);
        };
        if self.files.len() >= MAX_OPEN_FILES {
            tracing::warn!(device = device.path(), "nvsrv: too many files open");
            return (INVALID_FD, NvResult::FileOperationFailed);
        }
        let Ok(fd) = i32::try_from(self.next_fd) else {
            tracing::warn!(device = device.path(), "nvsrv: fds are used up");
            return (INVALID_FD, NvResult::FileOperationFailed);
        };
        self.files.insert(self.next_fd, device);
        self.next_fd += 1;
        (fd, NvResult::Success)
    }

    /// `Close` (`Module::Close`).
    pub fn close(&mut self, fd: i32) -> NvResult {
        match key(fd).map(|key| self.files.remove(&key)) {
            Err(result) => result,
            Ok(None) => NvResult::NotImplemented,
            Ok(Some(_)) => NvResult::Success,
        }
    }

    /// `QueryEvent` (`Module::QueryEvent`): no device has events in this slice (module docs).
    pub fn query_event(&self, fd: i32, event: u32) -> NvResult {
        match self.file(fd) {
            Err(result) => result,
            Ok(device) => {
                tracing::warn!(
                    device = device.path(),
                    event,
                    "nvsrv: an event no device has"
                );
                NvResult::BadParameter
            }
        }
    }

    /// An ioctl on `fd` with argument `input`: the fd checks, then the device, then the policy for
    /// an ioctl the device does not know (module docs, "Ioctls").
    pub fn ioctl(
        &mut self,
        policy: UnknownPolicy,
        variant: IoctlVariant,
        fd: i32,
        ioctl: Ioctl,
        input: &[u8],
    ) -> IoctlAnswer {
        let device = match self.file(fd) {
            Ok(device) => device,
            Err(result) => return IoctlAnswer::only(result),
        };
        let known = match (device, variant) {
            (Device::Nvmap, IoctlVariant::Ioctl) => self.nvmap.ioctl(ioctl, input),
            _ => None,
        };
        if let Some((result, output)) = known {
            return IoctlAnswer { result, output };
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
            UnknownPolicy::AutoStub => IoctlAnswer {
                result: NvResult::Success,
                output: vec![0; usize::try_from(ioctl.size()).unwrap_or_default()],
            },
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
        let (fd, result) = cx.family::<NvsrvState>()?.open(&path);
        reply.payload().write_i32(fd).write_u32(result.raw());
        reply.ok()
    }

    /// `Ioctl` (1).
    fn ioctl(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        run_ioctl(IoctlVariant::Ioctl, cx, req, reply)
    }

    /// `Ioctl2` (11): its inline input (in buffer 1) is not read (module docs).
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
        let result = cx.family::<NvsrvState>()?.close(fd);
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

    /// `QueryEvent` (4). in: `s32` fd, `u32` event id. out: the result.
    fn query_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let fd = payload.read_i32()?;
        let event = payload.read_u32()?;
        let result = cx.family::<NvsrvState>()?.query_event(fd, event);
        reply.payload().write_u32(result.raw());
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
/// out buffer 0, the argument back if the ioctl's `is_out` bit is set; the result.
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
    let policy = cx.policy();
    let answer = cx
        .family::<NvsrvState>()?
        .ioctl(policy, variant, fd, ioctl, &input);
    if ioctl.is_out() {
        let mut out = req.out_buffer(cx, 0)?;
        let len = usize::try_from(out.len())
            .map_or(answer.output.len(), |len| len.min(answer.output.len()));
        out.write(answer.output.get(..len).unwrap_or_default())?;
    }
    reply.payload().write_u32(answer.result.raw());
    reply.ok()
}

/// The interfaces an nvsrv server's sessions can be.
#[must_use]
pub fn nvsrv_interfaces() -> Interfaces {
    Interfaces::new().with::<Nvsrv>()
}

/// The nvsrv server: every session an [`Nvsrv`], sharing one [`NvsrvState`]. `policy` answers
/// unknown commands and, inside `Ioctl`, unknown ioctls (module docs).
#[must_use]
pub fn nvsrv_server(policy: UnknownPolicy) -> Server {
    Server::new(NVSRV_NAME, policy)
        .with_root(|| Box::new(Nvsrv))
        .with_interfaces(nvsrv_interfaces())
        .with_family(NvsrvState::new())
}
