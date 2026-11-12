//! Out buffers written by a reply ([`Reply::write_buffer`]): when the bytes reach the caller's
//! memory, and what happens when they cannot. An interface of its own, so the toy service's table
//! is untouched.

use serde::{Deserialize, Serialize};
use starquake_ipc::{BUFFER_SIZE, BufferDescriptor, CommandType, RequestBuilder};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::fake::FakeKernel;
use super::super::result::{Fault, NOT_SUPPORTED};
use super::super::{
    Context, DeferredBuffer, Handled, Outcome, Reply, Request, Server, UnknownPolicy,
};
use super::{Answer, STALE, only, plain};
use crate::commands;

/// What `fill` writes.
const FILL: [u8; 3] = [0x11, 0x22, 0x33];
/// Where the caller's room is, and how much of it is mapped.
const ROOM: u64 = 0x0040_0000;
const MAPPED: usize = 8;

/// Writes into out buffer 0 from its reply.
#[derive(Default, Serialize, Deserialize)]
struct Filler {
    /// An object to hand the caller a handle to, with the bytes.
    object: Option<ObjectId>,
}

mod cmd {
    pub(super) const FILL: u32 = 0;
    pub(super) const FILL_AND_HANDLE: u32 = 1;
    pub(super) const FILL_AND_FLOOD: u32 = 2;
}

commands! {
    impl Filler {
        0 => fill,
        1 => fill_and_handle,
        2 => fill_and_flood,
    }
}

impl Filler {
    fn fill(_: &mut Self, _: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let buffer = req.deferred_out_buffer(0)?;
        reply.write_buffer(buffer, FILL.to_vec())?;
        reply.ok()
    }

    fn fill_and_handle(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::fill(this, cx, req, reply)?;
        if let Some(object) = this.object {
            reply.copy_handle(object);
        }
        reply.ok()
    }

    /// A payload larger than any message buffer, with the bytes.
    fn fill_and_flood(
        this: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::fill(this, cx, req, reply)?;
        reply.payload().write_bytes(&[0; BUFFER_SIZE]);
        reply.ok()
    }
}

struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

fn rig() -> Rig {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let object = kernel.new_object();
    let mut server = Server::new("filler", UnknownPolicy::Strict);
    let filler = Filler {
        object: Some(object),
    };
    server.open(session, Box::new(filler)).unwrap();
    kernel.map(ROOM, vec![STALE; MAPPED]);
    Rig {
        kernel,
        server,
        session,
    }
}

impl Rig {
    fn send(&mut self, command: u32, address: u64, size: u64) -> Answer {
        let room = BufferDescriptor {
            address,
            size,
            flags: 0,
        };
        let request = RequestBuilder::cmif(CommandType::Request, command).receive_buffer(room);
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        plain(&buffer)
    }

    fn room(&self) -> &[u8] {
        self.kernel.memory(ROOM, MAPPED).unwrap()
    }
}

#[test]
fn a_reply_writes_its_bytes_into_the_caller_s_buffer() {
    let mut rig = rig();
    // A buffer exactly the bytes' size.
    let reply = rig.send(cmd::FILL, ROOM, 3);
    assert_eq!(reply, only(ResultCode::SUCCESS));
    assert_eq!(
        rig.room(),
        [0x11, 0x22, 0x33, STALE, STALE, STALE, STALE, STALE]
    );
}

#[test]
fn more_bytes_than_the_buffer_holds_are_refused_and_nothing_is_written() {
    let mut rig = rig();
    let reply = rig.send(cmd::FILL, ROOM, 2);
    let overrun = Fault::BufferOverrun {
        offset: 0,
        needed: 3,
        len: 2,
    };
    assert_eq!(reply, only(overrun.result()));
    assert_eq!(rig.room(), [STALE; MAPPED]);

    let mut reply = Reply::default();
    let buffer = DeferredBuffer { address: 0, len: 2 };
    assert_eq!(
        reply.write_buffer(buffer, vec![1, 2, 3]).err(),
        Some(overrun)
    );
    assert!(reply.write_buffer(buffer, vec![1, 2]).is_ok());
    assert_eq!(reply.take_writes(), [(buffer, vec![1, 2])]);
}

#[test]
fn a_write_the_kernel_refuses_is_its_error_and_no_handle_is_added() {
    let mut rig = rig();
    // The buffer starts two bytes before the end of the mapped room.
    let address = ROOM + u64::try_from(MAPPED).unwrap() - 2;
    let reply = rig.send(cmd::FILL_AND_HANDLE, address, 8);
    assert_eq!(reply, only(KernelError::InvalidCurrentMemory.code()));
    assert_eq!(rig.kernel.handle_count(), 0);
    assert_eq!(rig.room(), [STALE; MAPPED]);
    // With room for the bytes, both arrive.
    let reply = rig.send(cmd::FILL_AND_HANDLE, ROOM, 8);
    assert_eq!(
        (reply.result, reply.copies.len()),
        (ResultCode::SUCCESS.raw(), 1)
    );
    assert_eq!(rig.kernel.handle_count(), 1);
}

#[test]
fn a_reply_too_large_for_the_message_writes_no_bytes() {
    let mut rig = rig();
    let reply = rig.send(cmd::FILL_AND_FLOOD, ROOM, 8);
    assert_eq!(reply, only(NOT_SUPPORTED));
    assert_eq!(rig.room(), [STALE; MAPPED]);
}
