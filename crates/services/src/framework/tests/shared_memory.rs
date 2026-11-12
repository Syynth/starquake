//! A service that makes shared memory and hands the caller a handle to it: the pages it writes
//! are the ones behind the handle, and the kernel's refusals reach the caller as the reply's
//! result.

use serde::{Deserialize, Serialize};
use starquake_ipc::RequestBuilder;
use starquake_kernel::handle::Handle;
use starquake_kernel::memory::{SharedPages, SvcPermission, shared_memory_size};
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::ServiceId;

use super::super::fake::FakeKernel;
use super::super::{Context, Kernel, Outcome, Reply, Request, Server, UnknownPolicy};
use super::{built, only, payload, plain, request};
use crate::commands;

/// The byte the service writes at the end of the pages it makes.
const MARK: u8 = 0x5A;
/// Its command: in `u64 size, u32 permission`; out a copy handle.
const MAKE: u32 = 0;

/// A service that keeps the shared memory it made. The pages are not part of a snapshot.
#[derive(Default, Serialize, Deserialize)]
struct Pad {
    #[serde(skip)]
    pages: Vec<SharedPages>,
}

commands! {
    impl Pad {
        0 => make,
    }
}

impl Pad {
    /// in: `u64 size, u32 permission`. Makes the shared memory, marks its last byte, and
    /// replies with a copy handle to it.
    fn make(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut payload = req.payload();
        let size = payload.read_u64()?;
        let permission = payload.read_u32()?;
        let permission =
            SvcPermission::from_raw(permission).ok_or(KernelError::InvalidNewMemoryPermission)?;
        let (object, pages) = cx.create_shared_memory(size, permission)?;
        let last = pages.size() - 1;
        pages
            .write(last, &[MARK])
            .map_err(|_| KernelError::InvalidCurrentMemory)?;
        self.pages.push(pages);
        reply.copy_handle(object);
        reply.ok()
    }
}

fn make(size: u64, permission: u32) -> RequestBuilder {
    request(MAKE).data(&payload(|w| {
        w.write_u64(size).write_u32(permission);
    }))
}

/// A server of [`Pad`]s with one open session, and the fake kernel it runs on.
fn rig() -> (Server, FakeKernel, starquake_kernel::id::ObjectId) {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = Server::new("pad", UnknownPolicy::Strict)
        .with_service_id(ServiceId(0x60))
        .with_root(|| Box::new(Pad::default()));
    server.open(session, Box::new(Pad::default())).unwrap();
    (server, kernel, session)
}

fn send(
    server: &mut Server,
    kernel: &mut FakeKernel,
    session: starquake_kernel::id::ObjectId,
    request: &RequestBuilder,
) -> Vec<u8> {
    let mut buffer = built(request);
    server.handle(kernel, session, &mut buffer).unwrap();
    buffer
}

#[test]
fn the_caller_gets_a_handle_to_the_pages_the_service_wrote() {
    let (mut server, mut kernel, session) = rig();
    let page = shared_memory_size(1).unwrap();
    let read = u32::from(SvcPermission::Read.bits());
    let reply = plain(&send(&mut server, &mut kernel, session, &make(page + 1, read)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.copies.len(), 1);
    let object = kernel
        .resolve_handle(Handle::from_raw(reply.copies[0]))
        .unwrap();
    let (pages, permission) = kernel.shared_memory(object).unwrap();
    // Rounded up to whole pages, mapped as asked, and marked by the service.
    assert_eq!((pages.size(), permission), (2 * page, SvcPermission::Read));
    let mut mark = [0];
    pages.read(2 * page - 1, &mut mark).unwrap();
    assert_eq!(mark, [MARK]);
}

#[test]
fn the_kernels_refusals_are_the_replys_result_and_give_no_handle() {
    let (mut server, mut kernel, session) = rig();
    let page = shared_memory_size(1).unwrap();
    for (size, permission, error) in [
        (
            0,
            SvcPermission::Read,
            KernelError::InvalidSize,
        ),
        (
            page,
            SvcPermission::None,
            KernelError::InvalidNewMemoryPermission,
        ),
        (
            page,
            SvcPermission::ReadExecute,
            KernelError::InvalidNewMemoryPermission,
        ),
    ] {
        let raw = u32::from(permission.bits());
        let reply = plain(&send(&mut server, &mut kernel, session, &make(size, raw)));
        assert_eq!(reply, only(error.code()), "{size:#x} {permission:?}");
    }
    assert_eq!(kernel.handle_count(), 0);
}
