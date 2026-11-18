//! The camera session the root's 602 hands out, with PR's reply shapes
//! (`docs/specs/graphics-server.md`, "Video codec sessions": "camera (601, 602): sessions with
//! PR's reply shapes, everything else logged"; port slice 15).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97
//! (`IGraphicsSupervisor::CreateCameraContext`, `ICameraContext`), GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project; "PR" below. NX2 has neither 601 nor 602. Every fact here is
//! PR's reading, marked read: PR says the camera module's handlers "were not located", so the
//! shapes come from the SDK's client stubs, and the meaning of every command but 3 is a
//! **guess**.
//!
//! # The root commands
//!
//! - **601** (`CreateCameraContext`) answers success and nothing else: the root's stub
//!   (`crate::root`). PR checks only the phase (`m_phase == Phase::Ready`), which the root logs
//!   and does not refuse (decision 7).
//! - **602** (`OpenCameraSession`) reads a `u64` device type (words 2–3). PR answers
//!   [`GSV_UNKNOWN_ARGUMENT`] for one above [`MAX_DEVICE_TYPE`] ("value above 1 is 0x280f3"),
//!   checked before the phase; here too, as the only check. Otherwise a [`CameraSession`] on
//!   the root's server, copied (the root's house rule for handles; PR moves it).
//!
//! # The session
//!
//! CMIF, as PR registers it (`RegisterHandlers`, not `RegisterHandlersTipc`):
//!
//! | Id | Command | In | Out |
//! |---|---|---|---|
//! | 3 | [`OPEN_DEVICE`] | a 16-byte name | `u64` [`DEVICE`] for `"Default"` or `"External"` (zero-padded); [`GSV_UNKNOWN_ARGUMENT`] for any other |
//! | 21 | [`COMMAND_21`] | `u64` | nothing |
//! | 4, 12, 14, 16, 18, 19, 24, 25, 29 | [`NO_OUTPUT`] | anything | nothing |
//! | 9, 10, 11, 17, 30 | [`ONE_OUTPUT`] | anything | one `u64` 0 |
//! | 13 | [`TWO_OUTPUTS`] | anything | two `u64` 0 |
//!
//! PR: "The camera module's ... static initializer ... builds a two-entry name table with the
//! keys "External" and "Default", both mapped to the value pair {2, 2}, so the u64 reply is
//! modelled as that value (0x200000002)". A name compares up to its first zero byte, as PR's
//! `strnlen`. Every other command is the server's unknown-command policy.
//!
//! The session keeps nothing, and nothing is recorded but the requests.

use serde::{Deserialize, Serialize};
use starquake_kernel::result::{MODULE_BITS, ResultCode};
use starquake_services::commands;
use starquake_services::framework::{Context, Outcome, Reply, Request};

use crate::native::GSV_MODULE;

#[cfg(test)]
mod tests;

/// `OpenDevice` (PR `{3, D<&ICameraContext::OpenDevice>, "OpenDevice"}`).
pub const OPEN_DEVICE: u32 = 3;
/// PR `{21, D<&ICameraContext::Command21>, "Command21"}`.
pub const COMMAND_21: u32 = 21;
/// The commands PR answers with no output (`Generic<0>`).
pub const NO_OUTPUT: [u32; 9] = [4, 12, 14, 16, 18, 19, 24, 25, 29];
/// The commands PR answers with one `u64` 0 (`Generic<1>`).
pub const ONE_OUTPUT: [u32; 5] = [9, 10, 11, 17, 30];
/// The command PR answers with two `u64` 0 (`Generic<2>`).
pub const TWO_OUTPUTS: [u32; 1] = [13];

/// The names 3 takes (PR: `device == "Default" || device == "External"`).
pub const DEVICE_NAMES: [&[u8]; 2] = [b"Default", b"External"];
/// How many bytes 3's name takes (PR `std::array<char, 16> name`).
pub const NAME_BYTES: usize = 16;
/// What 3 answers for a known name (PR `*out_device = 0x200000002ULL`).
pub const DEVICE: u64 = 0x2_0000_0002;
/// The largest device type 602 takes (PR `packet[2] <= 1`, with `!packet[3]`).
pub const MAX_DEVICE_TYPE: u64 = 1;
/// The description of [`GSV_UNKNOWN_ARGUMENT`] (PR `Result(static_cast<ErrorModule>(0xf3),
/// 0x140)`).
pub const UNKNOWN_ARGUMENT_DESCRIPTION: u32 = 0x140;
/// gsv result `0x280f3`: 602's device type above [`MAX_DEVICE_TYPE`], and 3's unknown name (PR).
/// What it means is a **guess** (PR also answers it for an unknown memory id in 1211).
pub const GSV_UNKNOWN_ARGUMENT: ResultCode =
    ResultCode::from_raw(GSV_MODULE | (UNKNOWN_ARGUMENT_DESCRIPTION << MODULE_BITS));

/// The camera session (module docs). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CameraSession;

commands! {
    impl CameraSession {
        OPEN_DEVICE => open_device,
        COMMAND_21 => command_21,
        4 => no_output,
        12 => no_output,
        14 => no_output,
        16 => no_output,
        18 => no_output,
        19 => no_output,
        24 => no_output,
        25 => no_output,
        29 => no_output,
        9 => one_output,
        10 => one_output,
        11 => one_output,
        17 => one_output,
        30 => one_output,
        13 => two_outputs,
    }
}

/// Whether `name`, up to its first zero byte, is one 3 takes.
#[must_use]
pub fn is_device_name(name: &[u8; NAME_BYTES]) -> bool {
    let end = name
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(NAME_BYTES);
    DEVICE_NAMES.contains(&&name[..end])
}

impl CameraSession {
    /// `OpenDevice` (3): in a 16-byte name, out `u64` [`DEVICE`].
    fn open_device(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let name: [u8; NAME_BYTES] = req.payload().read_array()?;
        if !is_device_name(&name) {
            tracing::info!("gsv: camera device name not known");
            return Err(GSV_UNKNOWN_ARGUMENT.into());
        }
        reply.payload().write_u64(DEVICE);
        reply.ok()
    }

    /// PR's `Command21`: in a `u64`, logged.
    fn command_21(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let value = req.payload().read_u64()?;
        tracing::debug!(value, "gsv: camera command 21");
        reply.ok()
    }

    /// Answers `outputs` `u64` zeros, whatever the request holds (PR `Generic<N>`).
    fn zeros(req: &Request<'_>, reply: &mut Reply, outputs: usize) -> Outcome {
        tracing::debug!(command = req.command_id(), outputs, "gsv: camera command");
        for _ in 0..outputs {
            reply.payload().write_u64(0);
        }
        reply.ok()
    }

    /// PR `Generic<0>`.
    fn no_output(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::zeros(req, reply, 0)
    }

    /// PR `Generic<1>`.
    fn one_output(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::zeros(req, reply, 1)
    }

    /// PR `Generic<2>`.
    fn two_outputs(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::zeros(req, reply, 2)
    }
}
