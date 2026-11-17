//! `/dev/nvhost-tsg-gpu`: timeslice groups, the channels bound to each and their subcontexts
//! (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv", the device table's
//! tsg-gpu row: rows 26 and 45, observed; port slice 10).
//!
//! Ported from Eden PR #4588 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project) @
//! e5fb3b97: `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_tsg_gpu.cpp` and
//! `nvhost_tsg_gpu.h` (`nvhost_tsg_gpu::Ioctl1`, `OnClose`), `native_gpu_context.h`
//! (`DecodeNativeTsgBinding`) and `nvhost_gpu.cpp` (`nvhost_gpu::BindToTsg`'s `Busy`, the
//! channel's half of a binding). NX2 has no such device: it serves the path with a stub that
//! answers every ioctl with the input echoed (`eden-nx2/src/core/hle/service/nvdrv/devices/`
//! `nx2_stub_device.h`), which the service framework forbids since row 96.
//!
//! # The ioctls
//!
//! `Ioctl` (1) only, by PR's exact numbers (PR compares `command.raw`); every other number, and
//! every number on `Ioctl2` and `Ioctl3`, goes to the server's unknown-ioctl policy (PR answers
//! them `NotImplemented`, the strict policy's answer).
//!
//! | Number | Ioctl | Argument |
//! |---|---|---|
//! | [`SET_TIMESLICE`] `0xC0045408` | `NVGPU_TSG_IOCTL_SET_TIMESLICE` | `u32` microseconds |
//! | [`BIND_CHANNEL`] `0xC0045401` | bind a channel | `s32` channel fd |
//! | [`UNBIND_CHANNEL`] `0xC0045402` | unbind it | `s32` channel fd |
//! | [`BIND_CHANNEL_EX`] `0xC018540B` | bind with a subcontext | `s32` channel fd, `u32` subcontext id, 16 bytes of zero |
//!
//! **Checks**, PR's in its order; the first that fails answers, and nothing is written back:
//!
//! - `SET_TIMESLICE`: the in buffer or out buffer 0 not exactly 4 bytes, or a timeslice of 0:
//!   `BadParameter`. Otherwise the value is kept (the guest scheduler is not time-sliced, so
//!   nothing reads it) and the argument comes back as sent.
//! - The three bindings: the in buffer not exactly 4 bytes (24 for `BIND_CHANNEL_EX`); for
//!   `BIND_CHANNEL_EX`, a subcontext id at or past [`SUBCONTEXT_COUNT`] or a non-zero byte in
//!   the last 16; a negative channel fd; out buffer 0 not the in buffer's size: `BadParameter`.
//! - A bind: the channel already in this group, `Busy`; the group holding [`MAX_CHANNELS`],
//!   `InsufficientMemory`; the fd not an open `/dev/nvhost-gpu`, `BadParameter`; the channel in
//!   another group, `Busy` (the channel's own check, `BindToTsg`'s `if (tsg_group)`).
//! - An unbind: the channel not in this group, `BadParameter`.
//!
//! A binding that passes comes back as sent, with `Success`.
//!
//! # The state
//!
//! Each tsg-gpu fd's group: its timeslice and its channels by fd, each with its subcontext id.
//! Closing the group's fd unbinds them all (PR `OnClose`). Closing a channel's fd does not take it
//! out of its group, as in PR, where the group holds the channel open; its fd is never handed out
//! again, so the binding only ever answers an unbind. What binding a channel does to the channel
//! itself (PR hands it the group's macro context) is the channel's, port slice 11.
//!
//! # Where this differs from PR
//!
//! Each row is pinned by the test named, in `tsg_gpu/tests.rs`:
//!
//! | Case | PR | Here |
//! |---|---|---|
//! | A refused ioctl's out buffer | written with whatever the server's scratch buffer held | nothing written (`a_refused_binding_writes_nothing`) |
//! | The process check (`!process`, and `BindToTsg`'s `AccessDenied` when the channel belongs to another process) | made | not made: every fd of an nvsrv state is the one process's (`crate::nvsrv`, "Devices and fds") |

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ctrl_gpu::SUBCONTEXT_COUNT;
use super::{Device, IoctlAnswer, IoctlCall, IoctlVariant, NvResult};

#[cfg(test)]
mod tests;

/// `NVGPU_TSG_IOCTL_SET_TIMESLICE` (PR: `command.raw == 0xc0045408`).
pub const SET_TIMESLICE: u32 = 0xC004_5408;

/// Bind a channel (PR: `0xc0045401`, command 1).
pub const BIND_CHANNEL: u32 = 0xC004_5401;

/// Unbind a channel (PR: `0xc0045402`, command 2).
pub const UNBIND_CHANNEL: u32 = 0xC004_5402;

/// Bind a channel with a subcontext (PR: `0xc018540b`, command 0xB, `DecodeNativeTsgBinding`'s
/// `extended`).
pub const BIND_CHANNEL_EX: u32 = 0xC018_540B;

/// A timeslice's argument: one `u32` (PR: `input.size() != 4 || output.size() != 4`).
pub const TIMESLICE_BYTES: usize = 4;

/// A binding's argument: the channel fd (`DecodeNativeTsgBinding`: `input.size() != 4`).
pub const BINDING_BYTES: usize = 4;

/// An extended binding's argument: the channel fd, the subcontext id, 16 zero bytes
/// (`input.size() != 24`).
pub const BINDING_EX_BYTES: usize = 24;

/// Where an extended binding's subcontext id is (`std::memcpy(&id, input.data() + 4, ..)`).
pub const BINDING_EX_SUBCONTEXT: usize = 4;

/// Where an extended binding's zero bytes start (`input.begin() + 8`).
pub const BINDING_EX_ZEROS: usize = 8;

/// The most channels a group holds: PR `channels.size() >= 64` is `InsufficientMemory`.
pub const MAX_CHANNELS: usize = 64;

/// One timeslice group.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    /// The timeslice in microseconds, 0 until set.
    pub timeslice_us: u32,
    /// The bound channels: fd → subcontext id.
    pub channels: BTreeMap<u32, u32>,
}

/// Every tsg-gpu fd's group. Plain data, saved with the nvsrv state; deserialising goes through
/// `SavedTsgGpu`, which refuses a state the ioctls cannot reach ([`TsgStateError`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedTsgGpu")]
pub struct TsgGpu {
    groups: BTreeMap<u32, Group>,
}

/// The serialised form of [`TsgGpu`].
#[derive(Deserialize)]
struct SavedTsgGpu {
    groups: BTreeMap<u32, Group>,
}

/// Why a saved [`TsgGpu`] could not be read back. The checks run in this order, the first that
/// fails reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TsgStateError {
    /// A group holds more channels than a group can.
    #[error("timeslice group fd {fd} holds {count} channels, more than can be")]
    TooManyChannels {
        /// The group's fd.
        fd: u32,
        /// How many.
        count: usize,
    },
    /// A binding's subcontext id is one no binding can have.
    #[error("channel fd {channel} is bound with subcontext {subcontext}, which does not exist")]
    Subcontext {
        /// The channel's fd.
        channel: u32,
        /// The subcontext id.
        subcontext: u32,
    },
    /// A channel is bound in two groups.
    #[error("channel fd {channel} is bound in two timeslice groups")]
    TwoGroups {
        /// The channel's fd.
        channel: u32,
    },
}

impl TryFrom<SavedTsgGpu> for TsgGpu {
    type Error = TsgStateError;

    fn try_from(saved: SavedTsgGpu) -> Result<Self, Self::Error> {
        let SavedTsgGpu { groups } = saved;
        if let Some((&fd, group)) = groups
            .iter()
            .find(|(_, group)| group.channels.len() > MAX_CHANNELS)
        {
            let count = group.channels.len();
            return Err(TsgStateError::TooManyChannels { fd, count });
        }
        let bindings = groups.values().flat_map(|group| &group.channels);
        if let Some((&channel, &subcontext)) = bindings
            .clone()
            .find(|&(_, &subcontext)| subcontext >= SUBCONTEXT_COUNT)
        {
            return Err(TsgStateError::Subcontext {
                channel,
                subcontext,
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        if let Some((&channel, _)) = bindings.clone().find(|(channel, _)| !seen.insert(**channel)) {
            return Err(TsgStateError::TwoGroups { channel });
        }
        Ok(Self { groups })
    }
}

/// A binding, decoded (PR `NativeTsgBinding`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Binding {
    channel: u32,
    subcontext: u32,
}

/// PR's `DecodeNativeTsgBinding`: the exact size, then for the extended form the subcontext and
/// the zero bytes, then the fd's sign.
fn decode(input: &[u8], extended: bool) -> Option<Binding> {
    let size = if extended {
        BINDING_EX_BYTES
    } else {
        BINDING_BYTES
    };
    if input.len() != size {
        return None;
    }
    let word = |at: usize| {
        input
            .get(at..at + 4)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .map(u32::from_le_bytes)
    };
    let fd = word(0)?.cast_signed();
    let mut subcontext = 0;
    if extended {
        subcontext = word(BINDING_EX_SUBCONTEXT)?;
        let zeros = input.get(BINDING_EX_ZEROS..).unwrap_or_default();
        if subcontext >= SUBCONTEXT_COUNT || zeros.iter().any(|&byte| byte != 0) {
            return None;
        }
    }
    let channel = u32::try_from(fd).ok()?;
    Some(Binding {
        channel,
        subcontext,
    })
}

impl TsgGpu {
    /// No groups.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The group of tsg-gpu fd `fd`, if an ioctl made one.
    #[must_use]
    pub fn group(&self, fd: u32) -> Option<&Group> {
        self.groups.get(&fd)
    }

    /// The groups' fds, in order.
    pub fn fds(&self) -> impl Iterator<Item = u32> + '_ {
        self.groups.keys().copied()
    }

    /// Every bound channel's fd, in group order.
    pub fn channels(&self) -> impl Iterator<Item = u32> + '_ {
        self.groups.values().flat_map(|group| group.channels.keys().copied())
    }

    /// Group fd `fd` was closed: its channels are unbound (PR `OnClose`).
    pub(super) fn close(&mut self, fd: u32) {
        self.groups.remove(&fd);
    }

    /// An ioctl on tsg-gpu fd `fd`, whose channel fds are checked against `files`; `None` for
    /// one the device does not know (module docs, "The ioctls").
    pub(super) fn ioctl(
        &mut self,
        fd: u32,
        call: &IoctlCall<'_>,
        files: &BTreeMap<u32, Device>,
    ) -> Option<IoctlAnswer> {
        if call.variant != IoctlVariant::Ioctl {
            return None;
        }
        let result = match call.ioctl.raw() {
            SET_TIMESLICE => self.set_timeslice(fd, call),
            BIND_CHANNEL => self.bind(fd, call, files, false),
            BIND_CHANNEL_EX => self.bind(fd, call, files, true),
            UNBIND_CHANNEL => self.unbind(fd, call),
            _ => return None,
        };
        Some(match result {
            NvResult::Success => IoctlAnswer::with(result, call.input.to_vec()),
            refused => IoctlAnswer::only(refused),
        })
    }

    /// `SET_TIMESLICE`.
    fn set_timeslice(&mut self, fd: u32, call: &IoctlCall<'_>) -> NvResult {
        let value = <[u8; TIMESLICE_BYTES]>::try_from(call.input).map(u32::from_le_bytes);
        match value {
            Ok(value) if call.out_len == TIMESLICE_BYTES && value != 0 => {
                self.groups.entry(fd).or_default().timeslice_us = value;
                NvResult::Success
            }
            _ => NvResult::BadParameter,
        }
    }

    /// `BIND_CHANNEL` or, if `extended`, `BIND_CHANNEL_EX`.
    fn bind(
        &mut self,
        fd: u32,
        call: &IoctlCall<'_>,
        files: &BTreeMap<u32, Device>,
        extended: bool,
    ) -> NvResult {
        let Some(binding) = decode(call.input, extended) else {
            return NvResult::BadParameter;
        };
        if call.out_len != call.input.len() {
            return NvResult::BadParameter;
        }
        let own = self.groups.get(&fd);
        if own.is_some_and(|group| group.channels.contains_key(&binding.channel)) {
            return NvResult::Busy;
        }
        if own.is_some_and(|group| group.channels.len() >= MAX_CHANNELS) {
            return NvResult::InsufficientMemory;
        }
        if files.get(&binding.channel) != Some(&Device::NvhostGpu) {
            return NvResult::BadParameter;
        }
        if self
            .groups
            .values()
            .any(|group| group.channels.contains_key(&binding.channel))
        {
            return NvResult::Busy;
        }
        self.groups
            .entry(fd)
            .or_default()
            .channels
            .insert(binding.channel, binding.subcontext);
        NvResult::Success
    }

    /// `UNBIND_CHANNEL`.
    fn unbind(&mut self, fd: u32, call: &IoctlCall<'_>) -> NvResult {
        let Some(binding) = decode(call.input, false) else {
            return NvResult::BadParameter;
        };
        if call.out_len != call.input.len() {
            return NvResult::BadParameter;
        }
        let removed = self
            .groups
            .get_mut(&fd)
            .and_then(|group| group.channels.remove(&binding.channel));
        if removed.is_some() {
            NvResult::Success
        } else {
            NvResult::BadParameter
        }
    }
}
