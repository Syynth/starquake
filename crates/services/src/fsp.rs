//! `fsp-srv`, the file-system service: a first subset (`docs/specs/services.md`, port plan
//! slice 6). A title opens the SD card, opens a file on it, reads it, asks its size and closes
//! it; it opens its own data storage (its `RomFS` image), reads it and asks its size; and it opens
//! its own data file system (the files of its `RomFS`) and reads a file on it as on the SD card.
//!
//! Ported from `eden/src/core/hle/service/filesystem/fsp/fsp_srv.cpp` (`FSP_SRV`),
//! `fs_i_filesystem.cpp` (`IFileSystem`), `fs_i_file.cpp` (`IFile`) and `fs_i_storage.cpp`
//! (`IStorage`), with the checks of
//! `eden/src/core/file_sys/fsa/fs_i_file.h` and `fs_i_filesystem.h` (`FileSys::Fsa`), the open
//! modes of `eden/src/core/file_sys/fs_filesystem.h` and the result codes of
//! `eden/src/core/file_sys/errors.h`, all @ cf654052 (GPL-3.0-or-later). The request layouts are
//! checked against the client, libnx (`libnx/nx/source/services/fs.c`, ISC). Paths are normalised
//! by [`path`], a port of Eden's `fs_path_utility.h`. `GetGlobalAccessLogMode` is ported from
//! `fsp_srv.cpp` and `fsp_srv.h` (`AccessLogMode`, `FSP_SRV::access_log_mode`) with
//! `eden/src/common/settings.h` (`enable_fs_access_log`) @ cf654052, which Eden-NX2 @ 50fa86bb
//! and Eden PR #4588 @ e5fb3b97 have unchanged (see "The access log mode").
//!
//! # Commands
//!
//! Ids and names are Eden's tables (`FSP_SRV::FSP_SRV`, `IFileSystem::IFileSystem`,
//! `IFile::IFile`). Every other command of the three interfaces is left out and answered by the
//! server's unknown-command policy. Closing a file or a file system is the framework's close
//! (libnx `fsFileClose` and `fsFsClose` close the object), which drops the interface.
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `fsp-srv` ([`FspSrv`]) | 1 | `SetCurrentProcess` | the caller's process id, attached by the kernel; a `u64` placeholder in the payload, not read | nothing |
//! | | 2 | `OpenDataFileSystemByCurrentProcess` | nothing | an [`IFileSystem`] |
//! | | 18 | `OpenSdCardFileSystem` | nothing | an [`IFileSystem`] |
//! | | 200 | `OpenDataStorageByCurrentProcess` | nothing | an [`IStorage`] |
//! | | 1005 | `GetGlobalAccessLogMode` | nothing | `u32` mode: [`GLOBAL_ACCESS_LOG_MODE`] |
//! | [`IFileSystem`] | 8 | `OpenFile` | `u32` open mode; in buffer 0 (pointer, an X descriptor): the path | an [`IFile`] |
//! | [`IFile`] | 0 | `Read` | `u32` option, 4 bytes of padding, `s64` offset, `s64` size; out buffer 0 (map-alias, a B descriptor): room for the bytes | `s64` bytes read, and the bytes in out buffer 0 |
//! | | 4 | `GetSize` | nothing | `s64` size |
//! | [`IStorage`] | 0 | `Read` | `s64` offset, `s64` size; out buffer 0 (map-alias, a B descriptor): room for the bytes | the bytes in out buffer 0 |
//! | | 4 | `GetSize` | nothing | `s64` size |
//!
//! Layouts, each the client's and the reference's:
//!
//! - `SetCurrentProcess`: libnx `_fsInitialize`, `serviceDispatchIn(&g_fsSrv, 1, pid_placeholder,
//!   .in_send_pid = true)`; Eden `SetCurrentProcess(ClientProcessId pid)`, whose id is the
//!   kernel's (`cmif_serialization.h`, `InProcessId`: `pid = ctx.GetPID()`, "TODO: validate
//!   against raw data value?").
//! - `OpenSdCardFileSystem`: libnx `fsOpenSdCardFileSystem` (`_fsCmdGetSession`, one out
//!   object); Eden `OpenSdCardFileSystem(OutInterface<IFileSystem>)`.
//! - `OpenDataFileSystemByCurrentProcess`: libnx `fsOpenDataFileSystemByCurrentProcess`
//!   (`_fsCmdGetSession(&g_fsSrv, &out->s, 2)`, one out object); Eden
//!   `OpenDataFileSystemByCurrentProcess(OutInterface<IFileSystem>)`.
//! - `OpenFile`: libnx `_fsFsOpenCommon` (`u32` flags, a `HipcPointer` in buffer of `FS_MAX_PATH`
//!   (0x301) bytes, one out object); Eden `OpenFile(OutInterface<IFile>, InLargeData<Sf::Path,
//!   BufferAttr_HipcPointer>, u32 mode)`, and `Sf::Path` is `char str[EntryNameLengthMax + 1]`
//!   (`eden/src/core/file_sys/fssrv/fssrv_sf_path.h`).
//! - `Read`: libnx `fsFileRead` (`struct { u32 option; u32 pad; s64 offset; u64 read_size; }`, a
//!   `HipcMapAlias` out buffer, an `s64` out); Eden `Read(ReadOption, Out<s64>, s64 offset,
//!   OutBuffer<BufferAttr_HipcMapAlias ...>, s64 size)`, where `ReadOption` is a `u32`
//!   (`eden/src/core/file_sys/fs_file.h`).
//! - `GetSize`: libnx `fsFileGetSize`; Eden `GetSize(Out<s64>)`.
//! - `OpenDataStorageByCurrentProcess`: libnx `fsOpenDataStorageByCurrentProcess`
//!   (`_fsCmdGetSession(&g_fsSrv, &out->s, 200)`, one out object); Eden
//!   `OpenDataStorageByCurrentProcess(OutInterface<IStorage>)`.
//! - `IStorage::Read`: libnx `fsStorageRead` (`struct { s64 offset; u64 read_size; }`, a
//!   `HipcMapAlias` out buffer, nothing out); Eden `Read(OutBuffer<BufferAttr_HipcMapAlias ...>,
//!   s64 offset, s64 length)`.
//! - `IStorage::GetSize`: libnx `fsStorageGetSize(FsStorage*, s64*)`; Eden `GetSize(Out<u64>)`
//!   (the same eight bytes for every size up to `i64::MAX`, the most a size answer may be).
//! - `GetGlobalAccessLogMode`: libnx `fsGetGlobalAccessLogMode` (`_fsObjectDispatchOut(&g_fsSrv,
//!   1005, *out_mode)`, one `u32` out); Eden `GetGlobalAccessLogMode(Out<AccessLogMode>)`, where
//!   `AccessLogMode` is an `enum class : u32`.
//!
//! # The access log mode
//!
//! `GetGlobalAccessLogMode` answers [`GLOBAL_ACCESS_LOG_MODE`], `AccessLogMode::None`: Eden's
//! `access_log_mode` as `FSP_SRV` starts with it under Eden's default settings
//! (`enable_fs_access_log` off). It asks the host nothing and keeps nothing
//! (`get_global_access_log_mode_answers_eden_s_default`). Eden's answers whatever
//! `SetGlobalAccessLogMode` (1004) last set; 1004 is not ported, so it goes to the server's
//! unknown-command policy and the mode never changes here
//! (`set_global_access_log_mode_is_left_out_so_the_mode_stays_the_default`). Under auto-stub a
//! title that sets the mode is answered success and then reads `None` back, where Eden would
//! answer the mode it set. Porting 1004 makes the mode a session's state, as Eden's is: Eden
//! registers `fsp-srv` with a factory (`eden/src/core/hle/service/filesystem/filesystem.cpp`,
//! `LoopProcess`), and its server manager calls it for every session it accepts
//! (`eden/src/core/hle/service/server_manager.cpp`, `Port::CreateHandler`), so each session has
//! its own `FSP_SRV` and its own mode.
//!
//! # The data storage
//!
//! `OpenDataStorageByCurrentProcess` asks the host whether the run has a data storage (a probe
//! of [`Mount::DataStorage`](host::Mount::DataStorage)); the run's title answers with its
//! `RomFS`, built as an image (`starquake-core`'s narrow title API). [`IStorage`] reads it and
//! asks its size through the same host channel, with an empty path: the storage is one file.
//! Its `Read` is deferred as `IFile`'s is (see "The host behind it"), and its reply carries no
//! count: the bytes go into the buffer, fewer at the end of the storage, as Eden's backend reads
//! them (`VfsFile::Read` returns the bytes there are).
//!
//! # The data file system
//!
//! `OpenDataFileSystemByCurrentProcess` is Eden's `FSP_SRV::OpenDataFileSystemByCurrentProcess`,
//! which extracts the current process's `RomFS` (`FileSys::ExtractRomFS`) and hands out an
//! `IFileSystem` over the files in it. Here it probes
//! [`Mount::DataFileSystem`](host::Mount::DataFileSystem), and the [`IFileSystem`] it hands out
//! is the SD card's, on that mount: `OpenFile`, `IFile::Read` and `IFile::GetSize` are the same
//! commands with the same checks, and the host finds each path in the image (in a run,
//! `starquake-core`'s `FspHost`, through the loader's `RomFS` reader). The image is the one the
//! data storage serves, so a file reads as the bytes the storage holds at the file's data.
//!
//! # The host behind it
//!
//! A file system is a host folder named in the run's configuration, read-only. This crate never
//! reads the host itself: every question goes through the injectable environment's host I/O
//! channel ([`host`]), which the server is given with
//! [`Server::with_env`](crate::framework::Server::with_env). Opening the SD card asks whether it
//! is there, opening a file and `GetSize` ask the file's size, and each waits for the answer on
//! the core thread: they read metadata, never the file's contents, but they are blocking host I/O
//! all the same (see "Where this departs from the service spec"). `Read` is the one command whose
//! cost grows with what the guest asks for, so it never blocks the handler: it takes a reply
//! token and hands the read to the server's executor ([`work`](crate::framework::work)), whose
//! job asks the host for the bytes, on the same channel but through the environment the executor
//! gives it, and whose finish completes the token with them. The bytes go into the caller's
//! buffer when the reply is delivered ([`Reply::write_buffer`]). In a run the executor is
//! `fsp-srv`'s workers ([`worker`]), which also say what is recorded of each read and how a
//! replay delivers the reads in the recorded order.
//!
//! The host memory one read needs is the bytes it returns: at most the size the guest asked for,
//! which is at most its buffer's size, and never more than the file holds from the offset on.
//!
//! Sessions hold plain data (a mount and a path), so they are saved with the server. Their bytes
//! are pinned by `an_open_file_s_snapshot_bytes_are_pinned`.
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `fsp/tests.rs` or the files beside it in
//! `fsp/tests/`:
//!
//! - **No SD card.** Eden's `OpenSdCardFileSystem` ignores the `ResultPortSdCardNoDevice` that
//!   `FileSystemController::OpenSDMC` returns when there is no SD card, and replies success with
//!   a file system over no directory. Here the reply is [`result::PORT_SD_CARD_NO_DEVICE`]
//!   (`without_an_sd_card_opening_it_fails_with_no_device`).
//! - **Process registration.** Eden's `SetCurrentProcess` answers
//!   `FileSystemController::OpenProcess`, which is `ResultTargetNotFound` for a process its
//!   loader did not register. starquake has no registrations, so every process id the kernel
//!   attaches is accepted (`set_current_process_accepts_the_callers_process`). A request without
//!   one leaves Eden's id at its default, 0 (`hle_ipc.h`: `u64 pid{}`, set only when the request
//!   asks for it), which no process has (`k_process.h`: `InitialProcessIdMin = 1`), so both
//!   answer [`result::TARGET_NOT_FOUND`]
//!   (`set_current_process_without_a_process_id_is_target_not_found`). Eden stores the id for its
//!   save-data and RomFS commands; none is in this subset, so nothing is kept.
//! - **Paths.** Eden's `IFileSystem` hands the path to `VfsDirectoryServiceWrapper::OpenFile`
//!   (`eden/src/core/hle/service/filesystem/filesystem.cpp`), which runs
//!   `Common::FS::SanitizePath` (which turns backslashes into separators and drops a `..` at the
//!   root) and strips leading separators, so a relative path, or one that climbs above the root,
//!   still names a file. Here the path goes through [`path::normalize`], as a RomFS lookup needs
//!   (`docs/proposals.md`): it must start with `/`, may not climb above the root, and may not
//!   hold `:*?<>|` (`a_relative_path_is_refused`, `a_path_above_the_root_is_unobtainable`,
//!   `a_path_with_an_invalid_character_is_refused`).
//!
//!   The open mode is checked before the path, which is Eden's order: `Fsa::IFileSystem::OpenFile`
//!   (`eden/src/core/file_sys/fsa/fs_i_filesystem.h`) checks the mode, then its `DoOpenFile`
//!   reaches `SanitizePath`. Before either, the framework checks that the request carries the
//!   one pointer (X) buffer `OpenFile` declares, as the console does before calling the command,
//!   so a request with a bad mode and no path is `INVALID_CMIF_REQUEST`
//!   (`the_declared_buffers_are_checked_before_the_open_mode`). The console normalises first (Atmosphère
//!   `fssrv_filesystem_interface_adapter.cpp`, `FileSystemInterfaceAdapter::OpenFile` calls
//!   `SetUpPath` before `m_base_fs->OpenFile`, which checks the mode; fact only), so a request
//!   with both a bad mode and a bad path is [`result::INVALID_OPEN_MODE`] here and a path error
//!   on the console (`open_file_checks_eden_s_modes_then_read_only`). Eden's order is kept
//!   because this module ports Eden's `IFileSystem`, and only a request with both faults at once
//!   can tell the orders apart. That no title sends one is **a guess**; a trace of a title that
//!   does would settle it.
//! - **The path's buffer.** Eden copies at most `sizeof(Sf::Path)` (0x301) bytes of the buffer
//!   into a zeroed path and then reads it as a C string, past its end if none of the 0x301 bytes
//!   is zero. Here the path ends there (`a_path_with_no_terminator_ends_at_its_struct`). A
//!   request with no X descriptor for the path, or with an A descriptor, is the framework's
//!   `INVALID_CMIF_REQUEST`, where Eden logs a failed assertion and reads an empty path, or
//!   ignores the A descriptor (`open_file_without_a_path_is_an_invalid_request`,
//!   `open_file_refuses_a_map_alias_path`).
//! - **Read-only.** Eden's SD card can be written. Here a mode that is not exactly read is
//!   [`result::INVALID_OPEN_MODE`], after Eden's own two mode checks, as the console's read-only
//!   file systems answer (Atmosphère `fs_read_only_filesystem.hpp`, `ReadOnlyFileSystem::DoOpenFile`:
//!   `(mode & OpenMode_All) == OpenMode_Read`; fact only)
//!   (`open_file_checks_eden_s_modes_then_read_only`).
//! - **Read's checks.** Eden's `IFile::Read` reads into its buffer without comparing the size
//!   with it, and `Fsa::IFile::Read` answers a size of zero before it looks at the offset. Here
//!   the console's checks come first (Atmosphère `fssrv_filesystem_interface_adapter.cpp`,
//!   `FileInterfaceAdapter::Read`, fact only): a negative offset is
//!   [`result::INVALID_OFFSET`], then a negative size, or one larger than the buffer, is
//!   [`result::INVALID_SIZE`]; then Eden's: a size of zero answers at once with 0 bytes, and an
//!   offset and size whose sum overflows an `s64` is [`result::OUT_OF_RANGE`]
//!   (`read_checks_the_offset_before_the_size`, `read_checks_size_against_the_buffer`,
//!   `a_read_of_nothing_answers_at_once`, `a_read_whose_end_overflows_is_out_of_range`).
//! - **Host failures.** Eden's file system cannot fail to read a file it opened. A host failure
//!   here (a file that went away, an I/O error) is [`result::PATH_NOT_FOUND`] for a missing file
//!   and [`result::HOST_IO_FAILED`] otherwise
//!   (`a_host_failure_during_a_read_fails_the_read`). A host failure code with no name is
//!   [`result::HOST_IO_FAILED`] for `OpenFile` and for `Read`
//!   (`a_broken_host_fails_open_file_and_read`). A size answer that is not exactly 8 bytes, or
//!   that is over `i64::MAX`, is [`result::HOST_IO_FAILED`] for `OpenFile` and `GetSize`
//!   (`a_size_answer_of_other_than_eight_bytes_fails`, `a_size_over_s64_max_fails`), and so is a
//!   probe answered with any bytes (`a_probe_answered_with_bytes_opens_nothing`).
//!   [`result::HOST_IO_FAILED`] itself is a guess (see its docs): which code the reference has
//!   for an unreadable file waits for the owner.
//! - **The environment failing** is not a host failure and has no result code: it stops the run
//!   ([`halt`]; decision log, 2026-11-17, "Env failures"). The request is answered as one the
//!   server gave up on, `DEFERRED_REPLY_DROPPED`, which the guest never runs to see
//!   (`an_environment_failure_in_a_handler_stops_the_run` here, for each handler that reads the
//!   environment; `an_environment_failure_in_a_read_stops_the_run` in `fsp/tests/worker.rs`).
//!
//! - **No data storage.** Eden's `OpenDataStorageByCurrentProcess` answers `ResultUnknown`
//!   (`eden/src/core/hle/result.h`, `ResultUnknown(UINT32_MAX)`) when the current process has no
//!   `RomFS` ("No file system interface available!"). Here every answer to the probe but an
//!   empty one (no `RomFS`, a host failure, no handler) is that same [`result::UNKNOWN`]
//!   (`without_a_romfs_opening_the_data_storage_is_eden_s_unknown`), except the environment
//!   failing, which stops the run.
//! - **No data file system.** Eden's `OpenDataFileSystemByCurrentProcess` answers
//!   `ResultUnknown` when the current process has no `RomFS`, and when `ExtractRomFS` cannot
//!   read it ("Failed to extract RomFS for the current process!"). Here every answer to the
//!   probe of [`Mount::DataFileSystem`](host::Mount::DataFileSystem) but an empty one is
//!   [`result::UNKNOWN`] likewise, except the environment failing
//!   (`without_a_romfs_opening_the_data_file_system_is_eden_s_unknown`). Eden keeps the `RomFS`
//!   it opened (`FSP_SRV::romfs`); here every open asks the host again, so a session keeps
//!   nothing a snapshot would have to hold, and a second open can be refused where the first was
//!   not (`every_open_of_the_data_file_system_asks_the_host_again`).
//! - **A `RomFS` session restored into a run without one.** An [`IFileSystem`] or [`IFile`] on
//!   the data file system, and an [`IStorage`], are saved with the server, and restoring them
//!   does not check that the run has a `RomFS`. Eden's sessions hold the `RomFS` they were
//!   opened on, so they never meet a missing one. Here the host answers that the mount is not
//!   there ([`HostFailure::NoDevice`], or no host I/O handler at all), and `OpenFile`, `GetSize`
//!   and `Read` on these sessions answer [`result::UNKNOWN`], Eden's code for a process with no
//!   `RomFS`, not the SD card's [`result::PORT_SD_CARD_NO_DEVICE`]
//!   (`a_romfs_session_restored_without_a_romfs_answers_eden_s_unknown`).
//! - **`IStorage::Read`'s checks** are the console's (Atmosphère
//!   `fssrv_storage_interface_adapter.cpp`, `StorageInterfaceAdapter::Read`, fact only): a
//!   negative offset is [`result::INVALID_OFFSET`], then a negative size, or one larger than the
//!   buffer, is [`result::INVALID_SIZE`]. Eden checks the size before the offset, and not against
//!   the buffer, so a request with both a negative offset and a negative size is
//!   `INVALID_SIZE` there and `INVALID_OFFSET` here
//!   (`a_storage_read_checks_the_offset_before_the_size_and_the_buffer`). A size of zero answers
//!   at once with nothing read, as Eden's read of zero bytes ends.
//! - **`IStorage::Read` is stricter than the reference about an end past `s64`.** An offset and
//!   size whose sum overflows an `s64` is [`result::OUT_OF_RANGE`] here, the check `IFile`'s
//!   `Read` makes. Neither Eden's `IStorage::Read`
//!   (`eden/src/core/hle/service/filesystem/fsp/fs_i_storage.cpp`, which checks only
//!   `length >= 0` and `offset >= 0`, then reads) nor the console's
//!   `StorageInterfaceAdapter::Read` (Atmosphère, above; fact only) checks the sum, so there
//!   such a read succeeds with whatever its storage holds at the offset. Pinned by
//!   `a_storage_read_of_nothing_answers_at_once_and_one_that_overflows_is_out_of_range`.
//!   Eden's `IStorage::Read` also sleeps to hold reads to 40 MB/s; nothing here sleeps (a
//!   service never blocks), so reads take what the host takes.
//! - **Data storage failures.** A read of the storage that the host fails is
//!   [`result::HOST_IO_FAILED`], and a size answer that is not exactly 8 bytes is too, as for a
//!   file (`a_storage_read_the_host_fails_fails`). Eden's storage cannot fail.
//!
//! The framework's own rules apply as for every service: each command's buffers are declared in
//! its table row with the kind Eden's signature gives them, and a request whose descriptors do
//! not match is `INVALID_CMIF_REQUEST` before the command runs (`framework::buffers`): `Read`
//! takes one B descriptor (`read_refuses_a_pointer_buffer`), `OpenFile` one X descriptor, and
//! the other commands none. `IStorage`'s `Read` takes one B descriptor too
//! (`a_storage_read_refuses_a_pointer_buffer`).
//!
//! # Metadata on the calling thread
//!
//! `docs/specs/services.md`, "Concurrency", says methods never block, with one accepted
//! exception (approved by the owner, 2026-11-12): short host metadata lookups may run on the
//! calling thread. The three opens, `OpenFile` and the two `GetSize`s are those: each calls the
//! environment's host I/O on the core thread and waits for the answer. Behind a host folder
//! (`starquake-core`'s `HostFolder`) a probe is a `metadata` of the folder, and a size is two
//! `canonicalize`s, a `metadata` and a `File::open`, all made while the live environment holds
//! its lock; behind the title's `RomFS` a size is a lookup in its tables. Only `Read` is
//! deferred.
//!
//! Behind the title's `RomFS`, the first request for the data file system that an environment's
//! host gets also reads the image's whole directory and file tables (`starquake-core`'s
//! `FspHost`, through the loader's `RomFS` reader, which copies and checks them from the image
//! the loader built in memory, at most `romfs::build::MAX_METADATA_SIZE` bytes). For the
//! environment the handlers read, that request is the probe of the first
//! `OpenDataFileSystemByCurrentProcess`, so the tables are read on the core thread while the
//! live environment holds its lock. That is more than a short metadata lookup; whether it fits
//! the exception is for the owner to judge. (Each worker's environment reads the tables again,
//! at its own first read of the data file system, on its worker thread.)
//!
//! # Snapshots and reads in flight
//!
//! A snapshot taken while a `Read` is deferred saves the request as pending, but not the read
//! itself: its offset, length and buffer live only in the executor's job. A server restored
//! from it holds a pending token that nothing completes, and [`Server::reissue`] cannot restart
//! the read, so the guest waits for ever
//! (`a_snapshot_with_a_read_in_flight_restores_a_read_nothing_completes`).
//!
//! A restored [`IFile`]'s path is checked as it is decoded: it must be what [`path::normalize`]
//! makes of it, or the snapshot is refused
//! (`a_snapshot_with_a_path_that_is_not_normalised_is_refused`,
//! `a_saved_path_may_be_as_long_as_a_path_and_no_longer`).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_kernel::result::ResultCode;
use starquake_trace::Env;

use crate::commands;
use crate::framework::halt::{self, Halt};
use crate::framework::{
    Context, DeferredBuffer, Finish, Interfaces, Outcome, Reply, ReplyToken, Request, Server,
    UnknownPolicy,
};

pub mod host;
pub mod path;
pub mod worker;

#[cfg(test)]
mod tests;

use host::{Answer, HostFailure, HostRequest, Mount};

/// The name `fsp-srv` is registered under (`FSP_SRV::FSP_SRV`: `ServiceFramework{system_,
/// "fsp-srv"}`).
pub const NAME: &str = "fsp-srv";

/// The size of a path's buffer: `sizeof(Sf::Path)`, `EntryNameLengthMax + 1`
/// (`eden/src/core/file_sys/fssrv/fssrv_sf_path.h`; libnx `FS_MAX_PATH` is the same 0x301).
pub const PATH_BUFFER_SIZE: usize = path::ENTRY_NAME_LENGTH_MAX + 1;

/// What `GetGlobalAccessLogMode` answers: `AccessLogMode::None`, 0
/// (`eden/src/core/hle/service/filesystem/fsp/fsp_srv.h` @ cf654052: `enum class AccessLogMode
/// : u32 { None, Log, SdCard }`, and `FSP_SRV`'s `AccessLogMode access_log_mode =
/// AccessLogMode::None`). Eden's constructor sets `SdCard` instead when its
/// `enable_fs_access_log` setting is on, which it is not by default (`eden/src/common/settings.h`:
/// `Setting<bool> enable_fs_access_log{linkage, false, ..}`). The same in Eden-NX2 @ 50fa86bb and
/// Eden PR #4588 @ e5fb3b97. Module docs, "The access log mode".
pub const GLOBAL_ACCESS_LOG_MODE: u32 = 0;

/// The open-mode bits (`eden/src/core/file_sys/fs_filesystem.h`, `OpenMode`).
pub mod open_mode {
    /// `OpenMode::Read`.
    pub const READ: u32 = 1 << 0;
    /// `OpenMode::Write`.
    pub const WRITE: u32 = 1 << 1;
    /// `OpenMode::AllowAppend`.
    pub const ALLOW_APPEND: u32 = 1 << 2;
    /// `OpenMode::ReadWrite`.
    pub const READ_WRITE: u32 = READ | WRITE;
    /// `OpenMode::All`.
    pub const ALL: u32 = READ_WRITE | ALLOW_APPEND;
}

/// The result codes `fsp-srv` answers with, from `eden/src/core/file_sys/errors.h` (module
/// [`path::FS_MODULE`]), and the path normaliser's ([`path`]).
pub mod result {
    use starquake_kernel::result::ResultCode;

    use super::path::FS_MODULE;
    use crate::framework::result::code;

    pub use super::path::{
        DIRECTORY_UNOBTAINABLE, INVALID_CHARACTER, INVALID_PATH_FORMAT, TOO_LONG_PATH,
    };

    /// `ResultPathNotFound` (FS 1).
    pub const PATH_NOT_FOUND: ResultCode = code(FS_MODULE, 1);
    /// `ResultTargetNotFound` (FS 1002).
    pub const TARGET_NOT_FOUND: ResultCode = code(FS_MODULE, 1002);
    /// `ResultPortSdCardNoDevice` (FS 2001).
    pub const PORT_SD_CARD_NO_DEVICE: ResultCode = code(FS_MODULE, 2001);
    /// `ResultOutOfRange` (FS 3005).
    pub const OUT_OF_RANGE: ResultCode = code(FS_MODULE, 3005);
    /// The host could not answer a file request. **A guess**, starquake's own: no reference has a
    /// host behind its file systems. FS 5000 is the first code of the range Atmosphère names
    /// `Unexpected` (`vapours/results/fs_results.hpp`, fact only).
    pub const HOST_IO_FAILED: ResultCode = code(FS_MODULE, 5000);
    /// `ResultInvalidOffset` (FS 6061).
    pub const INVALID_OFFSET: ResultCode = code(FS_MODULE, 6061);
    /// `ResultInvalidSize` (FS 6062).
    pub const INVALID_SIZE: ResultCode = code(FS_MODULE, 6062);
    /// `ResultInvalidOpenMode` (FS 6072).
    pub const INVALID_OPEN_MODE: ResultCode = code(FS_MODULE, 6072);
    /// `ResultUnknown`: Eden's answer to `OpenDataStorageByCurrentProcess` when the process has
    /// no `RomFS` (`eden/src/core/hle/result.h`: `constexpr Result ResultUnknown(UINT32_MAX)`).
    pub const UNKNOWN: ResultCode = ResultCode::from_raw(u32::MAX);
}

/// The result a host answer that is not data stands for, on `mount`. The environment failing
/// stands for none: the run has stopped ([`halt`]), and the request is answered as one the server
/// gave up on, which the guest never runs to see.
fn failed(mount: Mount, answer: &Answer) -> ResultCode {
    match answer {
        Answer::Stopped => halt::STOPPED,
        Answer::Failed(HostFailure::NoDevice) | Answer::Unavailable => no_device(mount),
        Answer::Failed(HostFailure::NotFound | HostFailure::NotAFile) => result::PATH_NOT_FOUND,
        Answer::Failed(HostFailure::Io | HostFailure::BadRequest) | Answer::Broken => {
            result::HOST_IO_FAILED
        }
        Answer::Data(_) => {
            tracing::error!("the host answered a file request with data it should not have");
            result::HOST_IO_FAILED
        }
    }
}

/// The result for a `mount` the host does not have: the SD card's own code, and for the title's
/// `RomFS`, as its data storage or its data file system, Eden's [`result::UNKNOWN`], which is
/// what Eden answers a process with no `RomFS` (module docs, "No data storage", "No data file
/// system" and "A `RomFS` session restored into a run without one").
fn no_device(mount: Mount) -> ResultCode {
    match mount {
        Mount::SdCard => result::PORT_SD_CARD_NO_DEVICE,
        Mount::DataStorage | Mount::DataFileSystem => result::UNKNOWN,
    }
}

/// The size of the file at `path`, from the host.
fn file_size(env: &dyn Env, mount: Mount, path: &[u8]) -> Result<i64, ResultCode> {
    let path = path.to_vec();
    let answer = host::ask(env, &HostRequest::Size { mount, path });
    let Answer::Data(bytes) = &answer else {
        return Err(failed(mount, &answer));
    };
    let size = <[u8; 8]>::try_from(bytes.as_slice()).map(u64::from_le_bytes);
    let size = size.ok().and_then(|size| i64::try_from(size).ok());
    size.ok_or_else(|| {
        tracing::error!(
            len = bytes.len(),
            "the host answered a size request with a bad size"
        );
        result::HOST_IO_FAILED
    })
}

/// The root interface of an `fsp-srv` session (`FSP_SRV`). It keeps nothing: see the module
/// docs, "Process registration".
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FspSrv;

/// The ids of the commands of this subset, per interface (Eden's tables: `FSP_SRV::FSP_SRV`,
/// `IFileSystem::IFileSystem`, `IFile::IFile`).
pub mod command {
    /// [`FspSrv`](super::FspSrv)'s commands.
    pub mod fsp_srv {
        /// `SetCurrentProcess` (`{1, D<&FSP_SRV::SetCurrentProcess>, ..}`).
        pub const SET_CURRENT_PROCESS: u32 = 1;
        /// `OpenDataFileSystemByCurrentProcess`
        /// (`{2, D<&FSP_SRV::OpenDataFileSystemByCurrentProcess>, ..}`).
        pub const OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS: u32 = 2;
        /// `OpenSdCardFileSystem` (`{18, D<&FSP_SRV::OpenSdCardFileSystem>, ..}`).
        pub const OPEN_SD_CARD_FILE_SYSTEM: u32 = 18;
        /// `OpenDataStorageByCurrentProcess`
        /// (`{200, D<&FSP_SRV::OpenDataStorageByCurrentProcess>, ..}`).
        pub const OPEN_DATA_STORAGE_BY_CURRENT_PROCESS: u32 = 200;
        /// `GetGlobalAccessLogMode` (`{1005, D<&FSP_SRV::GetGlobalAccessLogMode>,
        /// "GetGlobalAccessLogMode"}`, the same row in Eden @ cf654052, Eden-NX2 @ 50fa86bb and
        /// Eden PR #4588 @ e5fb3b97); libnx `fsGetGlobalAccessLogMode` sends 1005.
        pub const GET_GLOBAL_ACCESS_LOG_MODE: u32 = 1005;
    }

    /// [`IFileSystem`](super::IFileSystem)'s commands.
    pub mod file_system {
        /// `OpenFile` (`{8, D<&IFileSystem::OpenFile>, ..}`).
        pub const OPEN_FILE: u32 = 8;
    }

    /// [`IFile`](super::IFile)'s commands.
    pub mod file {
        /// `Read` (`{0, D<&IFile::Read>, ..}`).
        pub const READ: u32 = 0;
        /// `GetSize` (`{4, D<&IFile::GetSize>, ..}`).
        pub const GET_SIZE: u32 = 4;
    }

    /// [`IStorage`](super::IStorage)'s commands (`fs_i_storage.cpp`, `IStorage::IStorage`).
    pub mod storage {
        /// `Read` (`{0, D<&IStorage::Read>, "Read"}`).
        pub const READ: u32 = 0;
        /// `GetSize` (`{4, D<&IStorage::GetSize>, "GetSize"}`).
        pub const GET_SIZE: u32 = 4;
    }
}

commands! {
    impl FspSrv {
        command::fsp_srv::SET_CURRENT_PROCESS => set_current_process,
        command::fsp_srv::OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS => open_data_file_system_by_current_process,
        command::fsp_srv::OPEN_SD_CARD_FILE_SYSTEM => open_sd_card_file_system,
        command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS => open_data_storage_by_current_process,
        command::fsp_srv::GET_GLOBAL_ACCESS_LOG_MODE => get_global_access_log_mode,
    }
}

impl FspSrv {
    /// `SetCurrentProcess`. in: the caller's process id (and a `u64` placeholder, not read).
    fn set_current_process(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if req.process_id().is_none() {
            return Err(result::TARGET_NOT_FOUND.into());
        }
        reply.ok()
    }

    /// `GetGlobalAccessLogMode`. out: `u32` mode, [`GLOBAL_ACCESS_LOG_MODE`] (module docs,
    /// "The access log mode").
    fn get_global_access_log_mode(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(GLOBAL_ACCESS_LOG_MODE);
        reply.ok()
    }

    /// `OpenSdCardFileSystem`. out: an [`IFileSystem`] on the SD card.
    fn open_sd_card_file_system(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mount = Mount::SdCard;
        let answer = host::ask(&*cx.env()?, &HostRequest::Probe { mount });
        match answer {
            Answer::Data(bytes) if bytes.is_empty() => {}
            answer => return Err(failed(mount, &answer).into()),
        }
        reply.push_session(Box::new(IFileSystem { mount }));
        reply.ok()
    }

    /// `OpenDataStorageByCurrentProcess`. out: an [`IStorage`] on the title's `RomFS` image.
    fn open_data_storage_by_current_process(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        probe_romfs(&*cx.env()?, Mount::DataStorage)?;
        reply.push_session(Box::new(IStorage));
        reply.ok()
    }

    /// `OpenDataFileSystemByCurrentProcess`. out: an [`IFileSystem`] on the files of the title's
    /// `RomFS` (module docs, "The data file system").
    fn open_data_file_system_by_current_process(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mount = Mount::DataFileSystem;
        probe_romfs(&*cx.env()?, mount)?;
        reply.push_session(Box::new(IFileSystem { mount }));
        reply.ok()
    }
}

/// Whether the host has `mount`, a way of reading the current process's `RomFS`: an empty answer
/// to a probe. Any other answer is Eden's [`result::UNKNOWN`] (module docs, "No data storage" and
/// "No data file system"), except the environment failing, which stops the run.
fn probe_romfs(env: &dyn Env, mount: Mount) -> Result<(), ResultCode> {
    match host::ask(env, &HostRequest::Probe { mount }) {
        Answer::Data(bytes) if bytes.is_empty() => Ok(()),
        Answer::Stopped => Err(failed(mount, &Answer::Stopped)),
        answer => {
            tracing::warn!(
                ?answer,
                ?mount,
                "the run has no RomFS for the current process to open this way"
            );
            Err(result::UNKNOWN)
        }
    }
}

/// A file system (`IFileSystem`), read-only: a host folder, or the files of the title's `RomFS`.
/// The default is the SD card.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IFileSystem {
    mount: Mount,
}

// The path is a pointer buffer: Eden's `InLargeData<Sf::Path, BufferAttr_HipcPointer>`.
commands! {
    impl IFileSystem {
        command::file_system::OPEN_FILE => open_file [In(Pointer)],
    }
}

/// Eden's two checks of an open mode (`Fsa::IFileSystem::OpenFile`), then the read-only file
/// system's (see the module docs, "Read-only").
fn check_open_mode(mode: u32) -> Result<(), ResultCode> {
    let eden = mode & open_mode::READ_WRITE != 0 && mode & !open_mode::ALL == 0;
    let read_only = mode & open_mode::ALL == open_mode::READ;
    if eden && read_only {
        Ok(())
    } else {
        Err(result::INVALID_OPEN_MODE)
    }
}

/// The form `OpenFile` gives a path before it asks the host, and that an [`IFile`] keeps: no
/// drive or Windows forms, and only the characters paths may hold.
fn normalise(raw: &[u8]) -> Result<Vec<u8>, path::PathError> {
    path::normalize(raw, PATH_BUFFER_SIZE, false, false, false)
}

/// The path in a request's in buffer 0: its first [`PATH_BUFFER_SIZE`] bytes at most. It ends at
/// its first zero byte, which is where [`path::normalize`] stops reading.
fn read_path(cx: &mut Context<'_>, req: &Request<'_>) -> Result<Vec<u8>, crate::framework::Fault> {
    let mut buffer = req.in_buffer(cx, 0)?;
    let len =
        usize::try_from(buffer.len()).map_or(PATH_BUFFER_SIZE, |len| len.min(PATH_BUFFER_SIZE));
    let mut bytes = vec![0; len];
    buffer.read(&mut bytes)?;
    Ok(bytes)
}

impl IFileSystem {
    /// `OpenFile`. in: `u32` mode; in buffer 0: the path. out: an [`IFile`].
    fn open_file(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mode = req.payload().read_u32()?;
        check_open_mode(mode)?;
        let raw = read_path(cx, req)?;
        let path = normalise(&raw).map_err(path::PathError::result)?;
        // The file must be there, and be a file.
        file_size(&*cx.env()?, this.mount, &path)?;
        reply.push_session(Box::new(IFile {
            mount: this.mount,
            path,
        }));
        reply.ok()
    }
}

/// An open file (`IFile`), read-only. The default has an empty path, which names no file: it is
/// for looking at the command table without opening a file.
///
/// It is decoded from a snapshot through `SavedFile`, which refuses a path that is not
/// normalised (see the module docs, "Snapshots and reads in flight").
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(try_from = "SavedFile")]
pub struct IFile {
    mount: Mount,
    /// The normalised path: what [`path::normalize`] makes of it is itself.
    path: Vec<u8>,
}

/// An [`IFile`] as a snapshot holds it, before its path is checked. The fields are [`IFile`]'s,
/// in its order, so the bytes are the same.
#[derive(Deserialize)]
struct SavedFile {
    mount: Mount,
    path: Vec<u8>,
}

/// A saved [`IFile`]'s path is not one `IFileSystem::open_file` could have made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the saved path of an open file is not a normalised path")]
pub struct NotNormalised;

impl TryFrom<SavedFile> for IFile {
    type Error = NotNormalised;

    fn try_from(saved: SavedFile) -> Result<Self, NotNormalised> {
        if !normalise(&saved.path).is_ok_and(|normal| normal == saved.path) {
            return Err(NotNormalised);
        }
        Ok(Self {
            mount: saved.mount,
            path: saved.path,
        })
    }
}

// The bytes go to a map-alias buffer: Eden's `OutBuffer<BufferAttr_HipcMapAlias |
// BufferAttr_HipcMapTransferAllowsNonSecure>` (the transfer mode is not checked; see
// `framework::buffers`).
commands! {
    impl IFile {
        command::file::READ => read [Out(MapAlias)],
        command::file::GET_SIZE => get_size,
    }
}

/// A read handed to the executor: what to ask the host, where the bytes go, and whether the
/// reply says how many were read (`IFile`'s does; `IStorage`'s does not).
struct ReadJob {
    mount: Mount,
    path: Vec<u8>,
    offset: u64,
    len: u64,
    buffer: DeferredBuffer,
    count: bool,
}

impl ReadJob {
    /// Asks the host through `env`, the environment the executor gives the job, and returns the
    /// step that completes the token with the answer ([`Finish`]).
    fn run(self, env: &dyn Env, token: ReplyToken) -> Finish {
        let answer = self.fetch(env);
        Box::new(move || match answer {
            Ok(reply) => token.complete(reply),
            Err(code) => token.fail(code),
        })
    }

    /// The reply: the bytes read as an `s64` (when the job says so), and the bytes for out
    /// buffer 0.
    fn fetch(self, env: &dyn Env) -> Result<Reply, ResultCode> {
        let request = HostRequest::Read {
            mount: self.mount,
            path: self.path,
            offset: self.offset,
            len: self.len,
        };
        let answer = host::ask(env, &request);
        let Answer::Data(bytes) = answer else {
            return Err(failed(self.mount, &answer));
        };
        let read = i64::try_from(bytes.len())
            .ok()
            .filter(|read| u64::try_from(*read).is_ok_and(|read| read <= self.len));
        let Some(read) = read else {
            tracing::error!(
                asked = self.len,
                got = bytes.len(),
                "the host answered a read with more bytes than were asked for"
            );
            return Err(result::HOST_IO_FAILED);
        };
        let mut reply = Reply::default();
        if self.count {
            reply.payload().write_i64(read);
        }
        reply
            .write_buffer(self.buffer, bytes)
            .map_err(|fault| fault.result())?;
        Ok(reply)
    }
}

/// A read of `size` bytes from `offset` of `path` on `mount` into out buffer 0: the checks
/// `IFile` and `IStorage` share (module docs, "Read's checks" and "`IStorage::Read`'s checks"),
/// then the read handed to the executor. `count` says whether the reply carries the number of
/// bytes read; a read of nothing answers at once.
fn deferred_read(
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
    target: (Mount, &[u8]),
    (offset, size): (i64, i64),
    count: bool,
) -> Outcome {
    let buffer = req.deferred_out_buffer(0)?;
    // The console's checks.
    let start = u64::try_from(offset).map_err(|_| result::INVALID_OFFSET)?;
    let len = u64::try_from(size).map_err(|_| result::INVALID_SIZE)?;
    if len > buffer.len() {
        return Err(result::INVALID_SIZE.into());
    }
    // Eden's (`Fsa::IFile::Read`: `CanAddWithoutOverflow<s64>(offset, size)`).
    if len == 0 {
        if count {
            reply.payload().write_i64(0);
        }
        return reply.ok();
    }
    if offset.checked_add(size).is_none() {
        return Err(result::OUT_OF_RANGE.into());
    }
    let executor = cx.executor()?;
    let token = cx.defer()?;
    let (mount, path) = target;
    let job = ReadJob {
        mount,
        path: path.to_vec(),
        offset: start,
        len,
        buffer,
        count,
    };
    executor.submit(Box::new(move |env: &dyn Env| job.run(env, token)));
    reply.deferred()
}

impl IFile {
    /// `Read`. in: `u32` option, padding, `s64` offset, `s64` size; out buffer 0. out: `s64`
    /// bytes read. Deferred: see the module docs, "The host behind it".
    fn read(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let _option = payload.read_u32()?;
        payload.align(8)?;
        let offset = payload.read_i64()?;
        let size = payload.read_i64()?;
        deferred_read(
            cx,
            req,
            reply,
            (this.mount, &this.path),
            (offset, size),
            true,
        )
    }

    /// `GetSize`. out: `s64` size.
    fn get_size(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let size = file_size(&*cx.env()?, this.mount, &this.path)?;
        reply.payload().write_i64(size);
        reply.ok()
    }
}

/// The current process's data storage (`IStorage`): its `RomFS` image, read-only. It keeps
/// nothing: there is one data storage, [`Mount::DataStorage`](host::Mount::DataStorage).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IStorage;

// The bytes go to a map-alias buffer: Eden's `OutBuffer<BufferAttr_HipcMapAlias |
// BufferAttr_HipcMapTransferAllowsNonSecure>`.
commands! {
    impl IStorage {
        command::storage::READ => read [Out(MapAlias)],
        command::storage::GET_SIZE => get_size,
    }
}

impl IStorage {
    /// `Read`. in: `s64` offset, `s64` size; out buffer 0. out: nothing but the bytes. Deferred:
    /// see the module docs, "The data storage".
    fn read(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut payload = req.payload();
        let offset = payload.read_i64()?;
        let size = payload.read_i64()?;
        deferred_read(
            cx,
            req,
            reply,
            (Mount::DataStorage, &[]),
            (offset, size),
            false,
        )
    }

    /// `GetSize`. out: `s64` size.
    fn get_size(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let size = file_size(&*cx.env()?, Mount::DataStorage, &[])?;
        reply.payload().write_i64(size);
        reply.ok()
    }
}

/// The interfaces an `fsp-srv` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<FspSrv>()
        .with::<IFileSystem>()
        .with::<IFile>()
        .with::<IStorage>()
}

/// An `fsp-srv` server whose host file systems are `env`'s ([`host`]), which tells `halt` when
/// `env` fails one of its handlers (the framework's one path for that, [`halt`]). Its reads run
/// on the executor it is given with [`Server::with_executor`] (in a run, [`worker::Workers`],
/// which tell a halt of their own),
/// through the environment that executor gives them; a server given none answers `Read` with
/// the framework's `NOT_SUPPORTED` ([`Context::executor`]).
#[must_use]
pub fn server(policy: UnknownPolicy, env: Arc<dyn Env>, halt: Arc<dyn Halt>) -> Server {
    Server::new(NAME, policy)
        .with_root(|| Box::new(FspSrv))
        .with_interfaces(interfaces())
        .with_env(env, halt)
}
