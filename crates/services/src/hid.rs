//! `hid`, the input service: as far as a libnx start-up asks, which is an applet resource and
//! its shared memory.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/service/hid/hid_server.cpp` (`IHidServer::CreateAppletResource`),
//! `applet_resource.cpp` (`IAppletResource`), `eden/src/hid_core/resource_manager.cpp`
//! (`ResourceManager::CreateAppletResource`, `GetSharedMemoryHandle`),
//! `eden/src/hid_core/resources/applet_resource.cpp` and `applet_resource.h` (`AppletResource`,
//! `AruidIndexMax`), `shared_memory_holder.cpp` (the memory's size and permissions) and
//! `eden/src/hid_core/hid_result.h` (the result codes). The request layouts are checked against
//! the client, libnx (`libnx/nx/source/services/hid.c`, ISC). The memory's layout is
//! [`shared_memory`].
//!
//! # Commands
//!
//! Ids and names are Eden's tables (`IHidServer::IHidServer`, `IAppletResource::IAppletResource`).
//! These are exactly the `hid` commands the startup fixture's trace sends
//! (`tests/homebrew/startup/reference/startup.trace`, lines 93 and 95); every other command is
//! left out and answered by the server's unknown-command policy.
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `hid` ([`IHidServer`]) | 0 | `CreateAppletResource` | the caller's process id, attached by the kernel; a `u64` applet resource user id in the payload, not read | an [`IAppletResource`] |
//! | [`IAppletResource`] | 0 | `GetSharedMemoryHandle` | nothing | copy handle: the applet's input shared memory |
//!
//! Layouts, each the client's and the reference's:
//!
//! - `CreateAppletResource`: libnx `_hidCreateAppletResource` (`serviceDispatchIn(srv, 0,
//!   AppletResourceUserId, .in_send_pid = true, .out_num_objects = 1)`); Eden
//!   `CreateAppletResource(OutInterface<IAppletResource>, ClientAppletResourceUserId aruid)`,
//!   whose id is the kernel's, not the payload's (`eden/src/core/hle/service/cmif_types.h`:
//!   `ClientAppletResourceUserId = ClientProcessId`; `cmif_serialization.h`, `InProcessId`:
//!   `pid = ctx.GetPID()`).
//! - `GetSharedMemoryHandle`: libnx `_hidGetSharedMemoryHandle` (`_hidCmdGetHandle`: one copied
//!   handle); Eden `OutCopyHandle<Kernel::KSharedMemory>`.
//!
//! # The applet resources
//!
//! Eden keeps a table of up to `AruidIndexMax` (0x20, [`ARUID_INDEX_MAX`]) applets, each with
//! its own input shared memory of [`shared_memory::SIZE`] bytes that processes map read-only
//! (`SharedMemoryHolder::Initialize`: user permission `Read`). Here the table is the server's
//! family state ([`AppletResources`]): an applet resource user id and its memory's kernel
//! object, which the server keeps for as long as it lives.
//!
//! `CreateAppletResource` answers success and an [`IAppletResource`] for the caller's id
//! whatever happened to the table, as Eden's does (`IHidServer::CreateAppletResource` logs the
//! resource manager's result and returns `R_SUCCEED()`). The memory is made the first time an
//! id asks, laid out as a fresh one ([`shared_memory::initial_image`]); an id that asks again
//! gets the same memory. When the table already holds [`ARUID_INDEX_MAX`] ids, a new one gets no
//! memory, and its `GetSharedMemoryHandle` answers [`ARUID_NOT_REGISTERED`]: Eden's answer for
//! an id with no row (`AppletResource::GetSharedMemoryHandle`, `index >= AruidIndexMax`).
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `hid/tests.rs`:
//!
//! - **Any applet resource user id gets a row.** Eden's applet manager registers an application's
//!   id when it launches it (`RegisterAppletResourceUserId`), and `CreateAppletResource` makes
//!   memory only for a registered id. starquake has no applet manager to register one, so
//!   `CreateAppletResource` registers the id itself, while the table has room
//!   (`a_full_table_gives_a_new_id_no_memory`). A request whose id Eden would find
//!   unregistered (any id, here) still answers success, as Eden's does.
//! - **No process id is refused.** Eden takes the id from the kernel whether or not the request
//!   asked for it. Here a request without it is answered [`INVALID_CMIF_REQUEST`]
//!   (`create_applet_resource_needs_the_process_id`), as `appletOE` answers its own
//!   (`crate::am`). Which code the console gives is a guess; a console trace would settle it.
//!   libnx always sends the id.
//! - **A failure to make the memory is the reply's.** Eden ignores the resource manager's result.
//!   Here an error from the kernel when the memory is made (a full object table) is the
//!   reply's result, and no row is kept
//!   (`a_kernel_failure_to_make_the_memory_is_the_reply_and_keeps_no_row`).
//! - **Nothing is ever sampled, and no controller is connected** ([`shared_memory`], "What a fresh
//!   one holds"). Eden connects a default controller and starts sampling the npad, six-axis,
//!   touch screen and gesture resources when an applet resource is created; there is no host
//!   input yet. A guest that reads the memory sees no controller and empty rings
//!   (`the_memory_is_a_fresh_input_shared_memory`).
//! - **Closing the last `IAppletResource` frees nothing.** Eden's destructor frees the id's row
//!   (`ResourceManager::FreeAppletResourceId`); the framework does not tell a service that a
//!   session closed, so the row and its memory stay for the server's life.
//!
//! [`INVALID_CMIF_REQUEST`]: crate::framework::result::INVALID_CMIF_REQUEST

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SvcPermission;
use starquake_kernel::result::{KernelError, ResultCode};

use crate::commands;
use crate::framework::result::{INVALID_CMIF_REQUEST, code};
use crate::framework::{
    Context, Fault, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy,
};

pub mod shared_memory;

#[cfg(test)]
mod tests;

/// The name `hid` is registered under (`eden/src/core/hle/service/hid/hid.cpp`, `LoopProcess`:
/// `RegisterNamedService("hid", ..)`).
pub const NAME: &str = "hid";

/// `ErrorModule::HID` (`eden/src/core/hle/result.h`).
pub const HID_MODULE: u32 = 202;

/// `ResultAruidNotRegistered` (`eden/src/hid_core/hid_result.h`: `{ErrorModule::HID, 1047}`):
/// the applet resource user id has no row.
pub const ARUID_NOT_REGISTERED: ResultCode = code(HID_MODULE, 1047);

/// How many applet resource user ids the table holds: `AruidIndexMax`
/// (`eden/src/hid_core/resources/applet_resource.h`).
pub const ARUID_INDEX_MAX: usize = 0x20;

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `hid`'s commands (`IHidServer::IHidServer`).
    pub mod hid_server {
        /// `CreateAppletResource` (`{0, C<&IHidServer::CreateAppletResource>, ..}`).
        pub const CREATE_APPLET_RESOURCE: u32 = 0;
    }

    /// `IAppletResource`'s commands (`IAppletResource::IAppletResource`).
    pub mod applet_resource {
        /// `GetSharedMemoryHandle` (`{0, C<&IAppletResource::GetSharedMemoryHandle>, ..}`).
        pub const GET_SHARED_MEMORY_HANDLE: u32 = 0;
    }
}

/// The root interface of a `hid` session (`IHidServer`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IHidServer;

/// An applet's resource (`IAppletResource`): what `CreateAppletResource` hands out, for one
/// applet resource user id.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IAppletResource {
    aruid: u64,
}

impl IAppletResource {
    /// The applet resource user id the resource is for.
    #[must_use]
    pub fn aruid(&self) -> u64 {
        self.aruid
    }
}

/// The `hid` server's family state: each applet resource user id that has asked, and its input
/// shared memory (module docs, "The applet resources").
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppletResources {
    memories: BTreeMap<u64, ObjectId>,
}

impl AppletResources {
    /// The input shared memory of `aruid`, if it has one.
    #[must_use]
    pub fn memory(&self, aruid: u64) -> Option<ObjectId> {
        self.memories.get(&aruid).copied()
    }

    /// How many ids have a row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Whether no id has a row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}

commands! {
    impl IHidServer {
        command::hid_server::CREATE_APPLET_RESOURCE => create_applet_resource,
    }
}

commands! {
    impl IAppletResource {
        command::applet_resource::GET_SHARED_MEMORY_HANDLE => get_shared_memory_handle,
    }
}

impl IHidServer {
    /// `CreateAppletResource`. in: the caller's process id (the applet resource user id).
    /// out: an [`IAppletResource`].
    fn create_applet_resource(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let Some(aruid) = req.process_id() else {
            return Err(INVALID_CMIF_REQUEST.into());
        };
        let resources = cx.family::<AppletResources>()?;
        let room = resources.memories.len() < ARUID_INDEX_MAX;
        if room && !resources.memories.contains_key(&aruid) {
            let memory = make_memory(cx)?;
            cx.family::<AppletResources>()?
                .memories
                .insert(aruid, memory);
        }
        reply.push_session(Box::new(IAppletResource { aruid }));
        reply.ok()
    }
}

/// Makes an input shared memory and lays it out as a fresh one.
///
/// The image is written in one write, which touches nothing unless it all fits; the kernel
/// rounds the size up, never down, so it does. Should it fail anyway, the object is not
/// returned: it stays the server's, as every object a server makes does.
fn make_memory(cx: &mut Context<'_>) -> Result<ObjectId, Fault> {
    let (memory, pages) = cx.create_shared_memory(shared_memory::SIZE, SvcPermission::Read)?;
    pages
        .write(0, &shared_memory::initial_image())
        .map_err(|fault| {
            tracing::error!(?fault, "the hid shared memory is smaller than its layout");
            Fault::Kernel(KernelError::InvalidState)
        })?;
    Ok(memory)
}

impl IAppletResource {
    /// `GetSharedMemoryHandle`. out: a copy handle to the applet's input shared memory.
    fn get_shared_memory_handle(
        &mut self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let memory = cx.family::<AppletResources>()?.memory(self.aruid);
        reply.copy_handle(memory.ok_or(Fault::Result(ARUID_NOT_REGISTERED))?);
        reply.ok()
    }
}

/// The interfaces a `hid` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IHidServer>()
        .with::<IAppletResource>()
}

/// The `hid` server.
#[must_use]
pub fn server(policy: UnknownPolicy) -> Server {
    Server::new(NAME, policy)
        .with_root(|| Box::new(IHidServer))
        .with_interfaces(interfaces())
        .with_family(AppletResources::default())
}
