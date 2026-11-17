//! The graphics server's calls on [`Context`], through the fake kernel: a hooked event whose
//! writable end the caller gets, the GPU-accessible range query, the device-share lock and
//! unlock, and a write into a transfer memory the request carries, each with the kernel's
//! refusals reaching the caller as the reply's result. A kernel that does not offer them answers
//! "not implemented".

use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use starquake_ipc::RequestBuilder;
use starquake_kernel::handle::Handle;
use starquake_kernel::id::{ObjectId, ProcessId, SlabId};
use starquake_kernel::ipc::EventHook;
use starquake_kernel::memory::testing::device_share_token;
use starquake_kernel::memory::{SharedPages, SvcPermission};
use starquake_kernel::result::{KernelError, ResultCode, SvcResult};
use starquake_trace::{EventKind, ServiceId};

use super::super::fake::FakeKernel;
use super::super::{Context, Kernel, Outcome, Reply, Request, Server, UnknownPolicy};
use super::{built, only, payload, plain, request};
use crate::commands;

/// Where the fake maps the caller's memory, and how much.
const MEMORY: u64 = 0x4000_0000;
const MEMORY_SIZE: u64 = 0x3000;

/// The commands. `MAKE_EVENT`: out a copy handle. `QUERY`: in `u64 address, u64 size`, out
/// `u8`. `LOCK`, `UNLOCK`: in `u64 address, u64 size`.
const MAKE_EVENT: u32 = 0;
const QUERY: u32 = 1;
const LOCK: u32 = 2;
const UNLOCK: u32 = 3;
/// `WRITE_LENT`: in copy handle 0, a transfer memory; `u64 offset`, `u32 word`. Writes the word
/// at the offset and replies the transfer memory's `u64 address`, `u64 size`.
const WRITE_LENT: u32 = 4;

/// What a hook heard: the events signalled, in order.
#[derive(Default, Clone)]
struct Heard(Arc<Mutex<Vec<ObjectId>>>);

impl Heard {
    fn events(&self) -> Vec<ObjectId> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl EventHook for Heard {
    fn signalled(&self, event: ObjectId) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

/// A service that uses the four calls. Its hook is not part of a snapshot, and nor is the
/// process its last lock's token named, which its unlock names with the request's range.
#[derive(Default, Serialize, Deserialize)]
struct Gpu {
    #[serde(skip)]
    heard: Heard,
    #[serde(skip)]
    process: Option<ProcessId>,
}

commands! {
    impl Gpu {
        MAKE_EVENT => make_event,
        QUERY => query,
        LOCK => lock,
        UNLOCK => unlock,
        WRITE_LENT => write_lent,
    }
}

/// A range from a request's payload: `u64 address, u64 size`.
fn range(req: &Request<'_>) -> Result<(u64, u64), super::super::Fault> {
    let mut payload = req.payload();
    Ok((payload.read_u64()?, payload.read_u64()?))
}

impl Gpu {
    fn make_event(&mut self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        let hook: Arc<dyn EventHook> = Arc::new(self.heard.clone());
        let (writable, _readable) = cx.create_hooked_event(hook)?;
        reply.copy_handle(writable);
        reply.ok()
    }

    fn query(_: &mut Self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let (address, size) = range(req)?;
        let accessible = cx.is_gpu_accessible(address, size)?;
        reply.payload().write_u8(u8::from(accessible));
        reply.ok()
    }

    fn lock(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let (address, size) = range(req)?;
        let token = cx.lock_device_share(address, size)?;
        assert_eq!((token.address(), token.size()), (address, size));
        self.process = Some(token.process());
        reply.ok()
    }

    /// Unlocks with a token for the request's range, naming the process the last lock's token
    /// named (or a made-up one before any lock), so a test can present any range.
    fn unlock(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let (address, size) = range(req)?;
        let process = self
            .process
            .unwrap_or_else(|| ProcessId::from_parts(u32::MAX, u32::MAX));
        cx.unlock_device_share(device_share_token(process, address, size))?;
        reply.ok()
    }
}

impl Gpu {
    fn write_lent(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let lent = req.copy_handle(0)?;
        let mut payload = req.payload();
        let (offset, word) = (payload.read_u64()?, payload.read_u32()?);
        let memory = cx.transfer_memory(lent)?;
        cx.write_transfer_memory(lent, offset, &word.to_le_bytes())?;
        reply
            .payload()
            .write_u64(memory.address())
            .write_u64(memory.size());
        reply.ok()
    }
}

/// `WRITE_LENT` with the handle `lent`, writing `word` at `offset`.
fn write_lent(lent: u32, offset: u64, word: u32) -> RequestBuilder {
    request(WRITE_LENT).copy_handle(lent).data(&payload(|w| {
        w.write_u64(offset).write_u32(word);
    }))
}

fn with_range(command: u32, address: u64, size: u64) -> RequestBuilder {
    request(command).data(&payload(|w| {
        w.write_u64(address).write_u64(size);
    }))
}

/// A server of [`Gpu`]s with one open session, whose hook is `heard`.
fn rig(heard: &Heard) -> (Server, ObjectId) {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = Server::new("gpu", UnknownPolicy::Strict).with_service_id(ServiceId(0x61));
    server
        .open(
            session,
            Box::new(Gpu {
                heard: heard.clone(),
                process: None,
            }),
        )
        .unwrap();
    (server, session)
}

fn send(
    server: &mut Server,
    kernel: &mut dyn Kernel,
    session: ObjectId,
    request: &RequestBuilder,
) -> super::Answer {
    let mut buffer = built(request);
    server.handle(kernel, session, &mut buffer).unwrap();
    plain(&buffer)
}

fn kernel() -> FakeKernel {
    let mut kernel = FakeKernel::new(0).unwrap();
    kernel.map(MEMORY, vec![0; usize::try_from(MEMORY_SIZE).unwrap()]);
    kernel
}

#[test]
fn the_caller_gets_the_writable_end_and_the_hook_hears_its_signals() {
    let heard = Heard::default();
    let (mut server, session) = rig(&heard);
    let mut kernel = kernel();
    let reply = send(&mut server, &mut kernel, session, &request(MAKE_EVENT));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.copies.len(), 1);
    let writable = kernel
        .resolve_handle(Handle::from_raw(reply.copies[0]))
        .unwrap();
    assert_eq!(heard.events(), []);
    // The guest signals twice: the hook hears each, with the writable end.
    kernel.guest_signal(writable).unwrap();
    kernel.guest_signal(writable).unwrap();
    assert_eq!(heard.events(), [writable, writable]);
    // A second event's signals are told to its own hook only, with its own id.
    let second = Heard::default();
    let other_session = kernel.new_object();
    server
        .open(
            other_session,
            Box::new(Gpu {
                heard: second.clone(),
                process: None,
            }),
        )
        .unwrap();
    let reply = send(
        &mut server,
        &mut kernel,
        other_session,
        &request(MAKE_EVENT),
    );
    let other = kernel
        .resolve_handle(Handle::from_raw(reply.copies[0]))
        .unwrap();
    assert_ne!(other, writable);
    kernel.guest_signal(other).unwrap();
    assert_eq!(second.events(), [other]);
    assert_eq!(heard.events(), [writable, writable]);
    // Anything that is not a hooked event's writable end runs nothing.
    let unhooked = kernel.new_object();
    assert_eq!(
        kernel.guest_signal(unhooked),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(heard.events().len(), 2);
}

#[test]
fn the_fakes_readable_end_counts_the_guests_signals() {
    let mut kernel = kernel();
    let heard = Heard::default();
    let (writable, readable) = kernel.create_hooked_event(Arc::new(heard.clone())).unwrap();
    assert_eq!(kernel.signals(readable), Some(0));
    kernel.guest_signal(writable).unwrap();
    assert_eq!(kernel.signals(readable), Some(1));
    // The writable end is not an event the fake counts signals of.
    assert_eq!(kernel.signals(writable), None);
}

#[test]
fn the_query_is_true_only_for_ranges_wholly_marked() {
    let heard = Heard::default();
    let (mut server, session) = rig(&heard);
    let mut kernel = kernel();
    kernel.mark_gpu_accessible(MEMORY, 0x2000);
    for (address, size, expected) in [
        (MEMORY, 0x2000, 1),
        (MEMORY + 0x1fff, 1, 1),
        (MEMORY, 0x2001, 0),
        (MEMORY - 1, 2, 0),
        (MEMORY, 0, 0),
        (u64::MAX, 2, 0),
    ] {
        let reply = send(
            &mut server,
            &mut kernel,
            session,
            &with_range(QUERY, address, size),
        );
        assert_eq!(reply.result, ResultCode::SUCCESS.raw());
        // One byte, padded to a word.
        assert_eq!(reply.data, [expected, 0, 0, 0], "{address:#x}+{size:#x}");
    }
}

#[test]
fn locks_nest_and_the_kernels_refusals_are_the_replys_result() {
    let heard = Heard::default();
    let (mut server, session) = rig(&heard);
    let mut kernel = kernel();
    let ok = only(ResultCode::SUCCESS);
    let invalid = only(KernelError::InvalidCurrentMemory.code());
    let mut call = |kernel: &mut FakeKernel, command, address, size| {
        send(
            &mut server,
            kernel,
            session,
            &with_range(command, address, size),
        )
    };
    assert_eq!(call(&mut kernel, LOCK, MEMORY, 0x1000), ok);
    assert_eq!(call(&mut kernel, LOCK, MEMORY, 0x1000), ok);
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), 2);
    // Memory the caller does not have, and a range no lock holds.
    assert_eq!(
        call(&mut kernel, LOCK, MEMORY + MEMORY_SIZE, 0x1000),
        invalid
    );
    assert_eq!(call(&mut kernel, UNLOCK, MEMORY, 0x2000), invalid);
    assert_eq!(call(&mut kernel, UNLOCK, MEMORY, 0x1000), ok);
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), 1);
    assert_eq!(call(&mut kernel, UNLOCK, MEMORY, 0x1000), ok);
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), 0);
    assert_eq!(call(&mut kernel, UNLOCK, MEMORY, 0x1000), invalid);
}

#[test]
fn the_fake_refuses_a_lock_past_the_largest_count() {
    let mut kernel = kernel();
    for _ in 0..u16::MAX {
        kernel.lock_device_share(MEMORY, 0x1000).unwrap();
    }
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), u16::MAX);
    assert_eq!(
        kernel.lock_device_share(MEMORY, 0x1000),
        Err(KernelError::OutOfResource)
    );
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), u16::MAX);
}

#[test]
fn the_fake_refuses_to_lock_an_empty_range() {
    // As the real kernel does: an empty range at mapped memory is not a range it can lock.
    let mut kernel = kernel();
    assert_eq!(
        kernel.lock_device_share(MEMORY, 0),
        Err(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(kernel.device_locks(MEMORY, 0), 0);
    // The same address with a size is lockable: the refusal is the size's.
    assert_eq!(
        kernel.lock_device_share(MEMORY, 0x1000),
        Ok(device_share_token(kernel.process(), MEMORY, 0x1000))
    );
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), 1);
}

#[test]
fn the_fakes_token_names_its_process_and_the_exact_range_and_another_process_s_is_refused() {
    let mut kernel = kernel();
    let token = kernel.lock_device_share(MEMORY, 0x1000).unwrap();
    assert_eq!(
        (token.process(), token.address(), token.size()),
        (kernel.process(), MEMORY, 0x1000)
    );
    // Another slot, and the fake's own slot in another generation.
    let other = ProcessId::from_parts(token.process().index().wrapping_add(1), 1);
    let stale = ProcessId::from_parts(
        token.process().index(),
        token.process().generation().wrapping_add(1),
    );
    for refused in [
        device_share_token(other, MEMORY, 0x1000),
        device_share_token(stale, MEMORY, 0x1000),
        device_share_token(kernel.process(), MEMORY, 0x2000),
        device_share_token(kernel.process(), MEMORY + 0x1000, 0x1000),
    ] {
        assert_eq!(
            kernel.unlock_device_share(refused),
            Err(KernelError::InvalidCurrentMemory)
        );
    }
    assert_eq!(kernel.device_locks(MEMORY, 0x1000), 1);
    assert_eq!(kernel.unlock_device_share(token), Ok(()));
    assert_eq!(
        kernel.unlock_device_share(token),
        Err(KernelError::InvalidCurrentMemory)
    );
}

/// A kernel that offers only what every kernel must: the graphics server's calls take their
/// defaults.
struct Bare(FakeKernel);

impl Kernel for Bare {
    fn process_id(&self) -> u64 {
        self.0.process_id()
    }

    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        self.0.resolve_handle(handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        self.0.insert_handle(object)
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        self.0.create_event()
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.0.signal_event(event)
    }

    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        self.0.create_shared_memory(size, permission)
    }

    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        self.0.create_session()
    }

    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        self.0.connect(service)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        self.0.read_memory(address, out)
    }

    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        self.0.write_memory(address, data)
    }

    fn record(&mut self, kind: EventKind) {
        self.0.record(kind);
    }
}

#[test]
fn a_kernel_that_does_not_offer_the_calls_answers_not_implemented() {
    let heard = Heard::default();
    let (mut server, session) = rig(&heard);
    let mut kernel = Bare(kernel());
    let object = kernel.0.lend(MEMORY, 0x1000, SvcPermission::None);
    let lent = kernel.0.insert_handle(object).unwrap().raw();
    let not_implemented = only(KernelError::NotImplemented.code());
    for message in [
        request(MAKE_EVENT),
        with_range(QUERY, MEMORY, 0x1000),
        with_range(LOCK, MEMORY, 0x1000),
        with_range(UNLOCK, MEMORY, 0x1000),
        write_lent(lent, 0, 1),
    ] {
        assert_eq!(
            send(&mut server, &mut kernel, session, &message),
            not_implemented
        );
    }
    assert_eq!(kernel.0.handle_count(), 1);
    assert_eq!(kernel.0.memory(MEMORY, 4), Some(&[0; 4][..]));
}

#[test]
fn a_server_writes_a_transfer_memory_the_request_carries_within_its_bounds() {
    const LENT: u64 = 0x10;
    let heard = Heard::default();
    let (mut server, session) = rig(&heard);
    let mut kernel = kernel();
    let object = kernel.lend(MEMORY + 0x100, LENT, SvcPermission::Read);
    let lent = kernel.insert_handle(object).unwrap().raw();
    // The last word inside is written, and the reply names the transfer memory.
    let reply = send(
        &mut server,
        &mut kernel,
        session,
        &write_lent(lent, LENT - 4, 0x0102_0304),
    );
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(
        reply.data,
        payload(|w| {
            w.write_u64(MEMORY + 0x100).write_u64(LENT);
        })
    );
    assert_eq!(
        kernel.memory(MEMORY + 0x100 + LENT - 4, 5),
        Some(&[4, 3, 2, 1, 0][..])
    );
    // One byte further reaches past the end: the kernel's error, nothing written.
    let reply = send(
        &mut server,
        &mut kernel,
        session,
        &write_lent(lent, LENT - 3, 0xFFFF_FFFF),
    );
    assert_eq!(reply, only(KernelError::InvalidCurrentMemory.code()));
    assert_eq!(kernel.memory(MEMORY + 0x100 + LENT, 1), Some(&[0][..]));
    // A handle to something that is not a transfer memory.
    let other = kernel.new_object();
    let other = kernel.insert_handle(other).unwrap().raw();
    let reply = send(&mut server, &mut kernel, session, &write_lent(other, 0, 1));
    assert_eq!(reply, only(KernelError::InvalidHandle.code()));
}
