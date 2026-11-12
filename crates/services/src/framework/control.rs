//! Control requests: the commands a session answers itself, whatever service is behind it.
//!
//! A message of type control (or control with context) is for the session, not the service. The
//! command ids and names are Eden's table (`eden/src/core/hle/service/sm/sm_controller.cpp`,
//! `Controller::Controller`). The payloads are what libnx, the client side, sends and reads back
//! (`libnx/nx/include/switch/sf/cmif.h`: `cmifConvertCurrentObjectToDomain`,
//! `cmifCopyFromCurrentDomain`, `cmifCloneCurrentObject`, `cmifQueryPointerBufferSize`,
//! `cmifCloneCurrentObjectEx`). Control replies never carry a domain header, on a domain or not:
//! control requests carry none (`eden/src/core/hle/service/ipc_helpers.h`, `ResponseBuilder`),
//! and libnx parses every control reply as a plain one (`cmifParseResponse(.., false, ..)`).
//!
//! | Id | Command | Payload in | Reply |
//! |---|---|---|---|
//! | 0 | convert current object to domain | none | `u32` object id of the session's interface |
//! | 1 | copy from current domain | `u32` object id | a moved session handle |
//! | 2 | clone current object | none | a moved session handle |
//! | 3 | query pointer buffer size | none | `u16` size |
//! | 4 | clone current object ex | `u32` tag, ignored | a moved session handle |
//!
//! - **Convert** follows Eden's `ConvertCurrentObjectToDomain`: the session becomes a domain and
//!   the reply holds object id 1. Eden converts after the reply is built; here the reply's
//!   framing is fixed before the conversion, which comes to the same bytes.
//! - **Clone** (2 and 4) follows Eden's `CloneCurrentObject`, which 4 calls without reading its
//!   argument: a new session that leads where this one does, returned as a moved handle
//!   (`Flags::AlwaysMoveHandles`), on a domain too.
//! - **Query** follows Eden's `QueryPointerBufferSize`: a `u16`. Eden clamps a per-process `u32`;
//!   here the size is the server's and a `u16` to begin with.
//! - **Copy from domain** has no handler in Eden, which lists the command by name only. The
//!   layout is libnx's `cmifCopyFromCurrentDomain`: a `u32` object id in, one moved handle out.
//!   What the handle leads to is a guess from the command's name: a new session on the object
//!   the id names, shared with the domain (closing either leaves the other working). A console
//!   trace, or a pinned Atmosphère, would confirm it.
//!
//! Left out: 5, set pointer buffer size, which Eden implements with the note "where does this
//! come from". It and any other id go to the server's unknown-command policy, as an unlisted
//! command of Eden's `Controller` does.
//!
//! Divergences from Eden, each pinned by a test in `framework/tests/control.rs`:
//!
//! - Converting a session that is already a domain: Eden logs a failed assertion, replies
//!   success with object id 1 and converts again, which resets the domain to hold only the
//!   session's first handler (`hle_ipc.h`, `ConvertToDomain`: `domain_handlers = {session_handler}`).
//!   Here the reply is [`NOT_SUPPORTED`](super::result::NOT_SUPPORTED) and the session and its
//!   objects are unchanged.
//! - Eden keeps one handler-and-domain state for a session and all its clones (`hle_ipc.h`,
//!   `SessionRequestManager`: "this state is shared when objects are cloned"), so converting one
//!   converts the others. Here a session and its clone lead to the same interface or the same
//!   domain, but converting one of two plain sessions leaves the other plain. Guess: this is what
//!   a console does, where the clone holds the object and not the session's state; a console
//!   trace would confirm it.

use starquake_ipc::CmifRequest;
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::commands::{Command, CommandTable, Protocol};
use super::kernel::Kernel;
use super::payload::Reader;
use super::result::{DOMAIN_OBJECT_NOT_FOUND, TARGET_NOT_DOMAIN};
use super::server::{Answer, Frame, Refusal, Server};
use super::targets::TargetId;

/// Control command 0: make the session a domain.
pub const CONVERT_CURRENT_OBJECT_TO_DOMAIN: u32 = 0;
/// Control command 1: a new session for one object of the session's domain.
pub const COPY_FROM_CURRENT_DOMAIN: u32 = 1;
/// Control command 2: a new session that leads where this one does.
pub const CLONE_CURRENT_OBJECT: u32 = 2;
/// Control command 3: the size of the server's pointer buffer.
pub const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
/// Control command 4: as [`CLONE_CURRENT_OBJECT`], with a tag the server ignores.
pub const CLONE_CURRENT_OBJECT_EX: u32 = 4;

/// The name control commands are logged under: Eden's name for its controller.
pub const CONTROL_INTERFACE: &str = "IpcController";

/// The control commands the framework answers, with Eden's names.
pub const CONTROL_COMMANDS: CommandTable = CommandTable {
    cmif: &[
        Command {
            id: CONVERT_CURRENT_OBJECT_TO_DOMAIN,
            name: "ConvertCurrentObjectToDomain",
            buffers: &[],
        },
        Command {
            id: COPY_FROM_CURRENT_DOMAIN,
            name: "CopyFromCurrentDomain",
            buffers: &[],
        },
        Command {
            id: CLONE_CURRENT_OBJECT,
            name: "CloneCurrentObject",
            buffers: &[],
        },
        Command {
            id: QUERY_POINTER_BUFFER_SIZE,
            name: "QueryPointerBufferSize",
            buffers: &[],
        },
        Command {
            id: CLONE_CURRENT_OBJECT_EX,
            name: "CloneCurrentObjectEx",
            buffers: &[],
        },
    ],
    tipc: &[],
};

impl Server {
    /// Answers a control request on `session`, which leads to `target`.
    pub(super) fn control(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        target: TargetId,
        body: &CmifRequest<'_>,
    ) -> Result<Answer, Refusal> {
        let mut payload = Reader::new(body.data);
        match body.header.command_id {
            CONVERT_CURRENT_OBJECT_TO_DOMAIN => {
                let object_id = self.convert_to_domain(session, target)?;
                Ok(Frame::Cmif.with_payload(&object_id.to_le_bytes()))
            }
            COPY_FROM_CURRENT_DOMAIN => {
                let object_id = payload.read_u32()?;
                let object = self.domain_object(target, object_id)?;
                Ok(moved(self.open_shared(kernel, object)?))
            }
            CLONE_CURRENT_OBJECT | CLONE_CURRENT_OBJECT_EX => {
                Ok(moved(self.open_shared(kernel, target)?))
            }
            QUERY_POINTER_BUFFER_SIZE => {
                Ok(Frame::Cmif.with_payload(&self.pointer_buffer_size.to_le_bytes()))
            }
            id => Ok(self.unknown(CONTROL_INTERFACE, Protocol::Cmif, id, Frame::Cmif)),
        }
    }

    /// The object `object_id` names in the domain `target` is.
    fn domain_object(&self, target: TargetId, object_id: u32) -> Result<TargetId, Refusal> {
        if !self.targets.is_domain(target)? {
            return Err(TARGET_NOT_DOMAIN.into());
        }
        let object = self.targets.domain(target)?.get(object_id);
        Ok(object.ok_or(DOMAIN_OBJECT_NOT_FOUND)?)
    }
}

/// A successful control reply that moves one handle to the caller.
fn moved(handle: Handle) -> Answer {
    let mut answer = Frame::Cmif.only(ResultCode::SUCCESS);
    answer.builder = answer.builder.move_handle(handle.raw());
    answer
}
