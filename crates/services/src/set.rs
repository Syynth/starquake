//! `set:sys`, the system settings service: the firmware version
//! (`docs/specs/services.md`, port plan slice 5).
//!
//! Ported from `eden/src/core/hle/service/set/system_settings_server.cpp`
//! (`ISystemSettingsServer::GetFirmwareVersion`, `GetFirmwareVersion2`, `GetFirmwareVersionImpl`)
//! and `settings_types.h` (`FirmwareVersionFormat`) @ cf654052 (GPL-3.0-or-later).
//!
//! # Commands
//!
//! Ids and names are Eden's table (`ISystemSettingsServer::ISystemSettingsServer`).
//!
//! | Id | Command | In | Out |
//! |---|---|---|---|
//! | 3 | `GetFirmwareVersion` | nothing | out buffer 0 (pointer, a C descriptor): a [`FirmwareVersion`], [`FIRMWARE_VERSION_SIZE`] bytes, with its minor revision zeroed |
//! | 4 | `GetFirmwareVersion2` | nothing | out buffer 0: the same, minor revision included |
//!
//! Every other command of `set:sys` is left out and answered by the server's unknown-command
//! policy. A homebrew program needs only these two: libnx calls one of them at start-up to learn
//! the system version (`libnx/nx/source/services/set.c`, `setsysGetFirmwareVersion`: command 4
//! from system version 3.0.0, command 3 before).
//!
//! The reply has no payload. The version goes into the caller's buffer: Eden declares it
//! `OutLargeData<FirmwareVersionFormat, BufferAttr_HipcPointer>`, and libnx sends a fixed-size
//! pointer buffer for it (`_setsysGetFirmwareVersionImpl`: `SfBufferAttr_FixedSize |
//! SfBufferAttr_HipcPointer | SfBufferAttr_Out`, `sizeof(SetSysFirmwareVersion)`), which reaches
//! the method as out buffer 0.
//!
//! Command 3 zeroes the minor revision (`GetFirmwareVersionImpl`: "If the command is
//! `GetFirmwareVersion` (as opposed to `GetFirmwareVersion2`), hardware will zero out the
//! `REVISION_MINOR` field").
//!
//! # Where the version comes from
//!
//! Eden reads it from the system version archive (system data `0100000000000809`, file `file`),
//! and synthesises that archive from constants when no firmware is installed
//! (`eden/src/core/file_sys/system_archive/system_version.cpp`). Here it is the `set` family's
//! state ([`Settings`]), given to [`sys_server`] by whoever puts the system together;
//! [`FirmwareVersion::eden_default`] is the version Eden synthesises. Reading a real archive is
//! the file system's work, not done here.
//!
//! # Where this differs from Eden
//!
//! Eden writes the buffer in `WriteOutArgument`
//! (`eden/src/core/hle/service/cmif_serialization.h`, the `OutLargeData` case), which for a
//! pointer buffer calls `HLERequestContext::WriteBufferC`
//! (`eden/src/core/hle/service/hle_ipc.cpp`). Each difference is pinned by the test named, in
//! `set/tests.rs`:
//!
//! - A buffer smaller than [`FIRMWARE_VERSION_SIZE`]: `WriteBufferC` logs, cuts the write down
//!   to the buffer and Eden replies success. Here nothing is written and the reply is
//!   `INVALID_CMIF_REQUEST`, the framework's rule for every out buffer (`framework::request`)
//!   (`a_buffer_too_small_gets_nothing_and_an_error`).
//! - No buffer at all: Eden's `ASSERT(ctx.CanWriteBuffer(..))` logs, `WriteBufferC` writes
//!   nothing, and the reply is success. Here the request fails the framework's check of the
//!   declared buffer and the reply is `INVALID_CMIF_REQUEST` (`a_request_without_a_buffer_fails`).
//! - A map-alias (B) buffer in place of the pointer buffer, or as well as it: `WriteBufferC`
//!   looks at C descriptors only, so Eden ignores the B descriptor, and writes nothing and
//!   replies success when there is no C descriptor. Here a B descriptor fails the framework's
//!   check (`framework::buffers`), as on the console, and the reply is `INVALID_CMIF_REQUEST`
//!   with nothing written (`a_map_alias_buffer_is_refused`). libnx never sends one.

use serde::{Deserialize, Serialize};

use crate::commands;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// The name `set:sys` is registered under (`ISystemSettingsServer`:
/// `ServiceFramework{system_, "set:sys"}`).
pub const SYS_NAME: &str = "set:sys";

/// `sizeof(FirmwareVersionFormat)` (`eden/src/core/hle/service/set/settings_types.h`:
/// `static_assert(sizeof(FirmwareVersionFormat) == 0x100)`).
pub const FIRMWARE_VERSION_SIZE: usize = 0x100;

/// Offsets and sizes of `FirmwareVersionFormat`'s fields
/// (`eden/src/core/hle/service/set/settings_types.h`; libnx's `SetSysFirmwareVersion` in
/// `libnx/nx/include/switch/services/set.h` has the same fields):
///
/// ```text
/// 0x00  u8 major
/// 0x01  u8 minor
/// 0x02  u8 micro
/// 0x03  1 byte of padding
/// 0x04  u8 revision_major
/// 0x05  u8 revision_minor
/// 0x06  2 bytes of padding
/// 0x08  char platform[0x20]
/// 0x28  u8 version_hash[0x40]
/// 0x68  char display_version[0x18]
/// 0x80  char display_title[0x80]
/// ```
pub mod layout {
    /// Offset of `major`.
    pub const MAJOR: usize = 0x00;
    /// Offset of `minor`.
    pub const MINOR: usize = 0x01;
    /// Offset of `micro`.
    pub const MICRO: usize = 0x02;
    /// Offset of `revision_major`.
    pub const REVISION_MAJOR: usize = 0x04;
    /// Offset of `revision_minor`.
    pub const REVISION_MINOR: usize = 0x05;
    /// Offset of `platform`.
    pub const PLATFORM: usize = 0x08;
    /// Size of `platform`.
    pub const PLATFORM_SIZE: usize = 0x20;
    /// Offset of `version_hash`.
    pub const VERSION_HASH: usize = 0x28;
    /// Size of `version_hash`.
    pub const VERSION_HASH_SIZE: usize = 0x40;
    /// Offset of `display_version`.
    pub const DISPLAY_VERSION: usize = 0x68;
    /// Size of `display_version`.
    pub const DISPLAY_VERSION_SIZE: usize = 0x18;
    /// Offset of `display_title`.
    pub const DISPLAY_TITLE: usize = 0x80;
    /// Size of `display_title`.
    pub const DISPLAY_TITLE_SIZE: usize = 0x80;
}

/// Why a set of fields is not a firmware version.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FirmwareVersionError {
    /// A text field does not fit its place with a terminating zero byte.
    #[error("{field} is {len} bytes, more than the {max} that fit its field with a terminator")]
    TooLong {
        /// The field's name.
        field: &'static str,
        /// The text's length in bytes.
        len: usize,
        /// The most that fits.
        max: usize,
    },
    /// A text field holds a zero byte, which would end it early for a reader.
    #[error("{field} holds a zero byte")]
    EmbeddedZero {
        /// The field's name.
        field: &'static str,
    },
}

/// The fields of a [`FirmwareVersion`], not yet checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirmwareVersionFields {
    /// System version, major part.
    pub major: u8,
    /// System version, minor part.
    pub minor: u8,
    /// System version, micro part.
    pub micro: u8,
    /// SDK revision, major part.
    pub revision_major: u8,
    /// SDK revision, minor part.
    pub revision_minor: u8,
    /// The platform (`"NX"`).
    pub platform: String,
    /// The version's hash, as text.
    pub version_hash: String,
    /// The version as shown to the user (`"23.0.0"`).
    pub display_version: String,
    /// The version's full title.
    pub display_title: String,
}

/// A system firmware version: what `GetFirmwareVersion` reports.
///
/// The text fields are checked when the value is made: each fits its field with at least one
/// zero byte after it and holds none itself, so a reader that takes a field as a C string gets
/// the whole text and nothing else. That rule is starquake's own, about values it is configured
/// with; the layout itself has no terminator requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "FirmwareVersionFields", into = "FirmwareVersionFields")]
pub struct FirmwareVersion(FirmwareVersionFields);

impl FirmwareVersion {
    /// The version with these fields, if they fit.
    pub fn new(fields: FirmwareVersionFields) -> Result<Self, FirmwareVersionError> {
        let texts = [
            ("platform", &fields.platform, layout::PLATFORM_SIZE),
            (
                "version_hash",
                &fields.version_hash,
                layout::VERSION_HASH_SIZE,
            ),
            (
                "display_version",
                &fields.display_version,
                layout::DISPLAY_VERSION_SIZE,
            ),
            (
                "display_title",
                &fields.display_title,
                layout::DISPLAY_TITLE_SIZE,
            ),
        ];
        for (field, text, size) in texts {
            let max = size.saturating_sub(1);
            if text.len() > max {
                let len = text.len();
                return Err(FirmwareVersionError::TooLong { field, len, max });
            }
            if text.bytes().any(|byte| byte == 0) {
                return Err(FirmwareVersionError::EmbeddedZero { field });
            }
        }
        Ok(Self(fields))
    }

    /// The version Eden reports when no firmware is installed: the constants of
    /// `eden/src/core/hle/api_version.h` (`HOS_VERSION_*`, `SDK_REVISION_*`, `PLATFORM_STRING`,
    /// `VERSION_HASH`, `DISPLAY_VERSION`, `DISPLAY_TITLE`), which
    /// `eden/src/core/file_sys/system_archive/system_version.cpp` writes into the version file.
    /// It is a Switch 1 system version; what a Switch 2 title expects here is not in any pinned
    /// reference.
    #[must_use]
    pub fn eden_default() -> Self {
        Self(FirmwareVersionFields {
            major: 23,
            minor: 0,
            micro: 0,
            revision_major: 1,
            revision_minor: 0,
            platform: "NX".to_owned(),
            version_hash: "c2663accb7490a6ccfe9bc4dfd120ca215490525".to_owned(),
            display_version: "23.0.0".to_owned(),
            display_title: "NintendoSDK Firmware for NX 23.0.0-4.0".to_owned(),
        })
    }

    /// The fields.
    #[must_use]
    pub fn fields(&self) -> &FirmwareVersionFields {
        &self.0
    }

    /// The version as `FirmwareVersionFormat` lays it out (see [`layout`]): padding and the
    /// rest of each text field are zero.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; FIRMWARE_VERSION_SIZE] {
        let fields = &self.0;
        let mut bytes = [0; FIRMWARE_VERSION_SIZE];
        let numbers = [
            (layout::MAJOR, fields.major),
            (layout::MINOR, fields.minor),
            (layout::MICRO, fields.micro),
            (layout::REVISION_MAJOR, fields.revision_major),
            (layout::REVISION_MINOR, fields.revision_minor),
        ];
        for (offset, value) in numbers {
            if let Some(slot) = bytes.get_mut(offset) {
                *slot = value;
            }
        }
        let texts = [
            (layout::PLATFORM, &fields.platform),
            (layout::VERSION_HASH, &fields.version_hash),
            (layout::DISPLAY_VERSION, &fields.display_version),
            (layout::DISPLAY_TITLE, &fields.display_title),
        ];
        for (offset, text) in texts {
            // The constructor checked that the text fits its field, so the zip takes it all.
            let field = bytes.iter_mut().skip(offset);
            for (slot, byte) in field.zip(text.bytes()) {
                *slot = byte;
            }
        }
        bytes
    }
}

impl TryFrom<FirmwareVersionFields> for FirmwareVersion {
    type Error = FirmwareVersionError;

    fn try_from(fields: FirmwareVersionFields) -> Result<Self, Self::Error> {
        Self::new(fields)
    }
}

impl From<FirmwareVersion> for FirmwareVersionFields {
    fn from(version: FirmwareVersion) -> Self {
        version.0
    }
}

/// The `set` family's state: the settings its services report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// The system's firmware version.
    pub firmware: FirmwareVersion,
}

/// The interface behind a `set:sys` session. It has no state of its own: what it reports is the
/// family's ([`Settings`]).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SetSys;

/// The ids of [`SetSys`]'s commands (`ISystemSettingsServer::ISystemSettingsServer`).
pub mod command {
    /// `GetFirmwareVersion` (`{3, C<&ISystemSettingsServer::GetFirmwareVersion>, ..}`).
    pub const GET_FIRMWARE_VERSION: u32 = 3;
    /// `GetFirmwareVersion2` (`{4, C<&ISystemSettingsServer::GetFirmwareVersion2>, ..}`).
    pub const GET_FIRMWARE_VERSION2: u32 = 4;
}

// Both write the version to a pointer buffer: Eden declares it
// `OutLargeData<FirmwareVersionFormat, BufferAttr_HipcPointer>` (module docs, "Commands").
commands! {
    impl SetSys {
        command::GET_FIRMWARE_VERSION => get_firmware_version [Out(Pointer)],
        command::GET_FIRMWARE_VERSION2 => get_firmware_version2 [Out(Pointer)],
    }
}

impl SetSys {
    /// `GetFirmwareVersion`. out buffer 0: `FirmwareVersionFormat`, `revision_minor` zeroed.
    fn get_firmware_version(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut version = cx.family::<Settings>()?.firmware.to_bytes();
        if let Some(revision_minor) = version.get_mut(layout::REVISION_MINOR) {
            *revision_minor = 0;
        }
        req.out_buffer(cx, 0)?.write(&version)?;
        reply.ok()
    }

    /// `GetFirmwareVersion2`. out buffer 0: `FirmwareVersionFormat`.
    fn get_firmware_version2(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let version = cx.family::<Settings>()?.firmware.to_bytes();
        req.out_buffer(cx, 0)?.write(&version)?;
        reply.ok()
    }
}

/// The interfaces a `set:sys` server's sessions can be.
#[must_use]
pub fn sys_interfaces() -> Interfaces {
    Interfaces::new().with::<SetSys>()
}

/// The `set:sys` server, reporting `settings`.
#[must_use]
pub fn sys_server(settings: Settings, policy: UnknownPolicy) -> Server {
    Server::new(SYS_NAME, policy)
        .with_root(|| Box::new(SetSys))
        .with_interfaces(sys_interfaces())
        .with_family(settings)
}
