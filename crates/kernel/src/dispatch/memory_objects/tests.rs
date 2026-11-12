//! The shared and transfer memory arms of the dispatcher, and a server's side of both through
//! [`IpcContext`]: which registers each call reads and writes, a shared memory a server makes
//! and the guest maps, a transfer memory the guest sends a server, and the pages a transfer
//! memory's last handle gives back as the kernel lock goes.

use std::sync::{Arc, Mutex, PoisonError};

use starquake_cpu::GuestRegs;
use starquake_ipc::{BUFFER_SIZE, CommandType, RequestBuilder, SpecialHeader};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, Perms};
use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId};

use super::*;
use crate::dispatch::{Caller, IpcContext, Record};
use crate::id::ObjectId;
use crate::ipc::{Handled, HandlerError, PortName, SessionHandler};
use crate::memory::fixture::{self, CODE, CODE_SIZE, DATA, SHARED};
use crate::memory::testing::{MapperCall, MapperLog};
use crate::memory::{
    MemoryAttribute, MemoryPermission, MemoryState, SharedPages, SvcPermission, TransferMemory,
};
use crate::object::Object;
use crate::process::{Console, ProcessDescription};
use crate::result::{KernelError, SvcResult};
use crate::svc::DebugSink;

/// The guest thread of these tests, as the event stream names it.
const THREAD: ThreadId = ThreadId(0xA1);
/// Its message buffer: the data's second page. The first is what the tests lend.
const TLS: u64 = DATA + PAGE_SIZE;
/// Where the tests put the port name: the data's last page.
const NAMES: u64 = DATA + 2 * PAGE_SIZE;
/// The test server's service number.
const SERVICE: ServiceId = ServiceId(0x51);
/// Pages in the shared memory the server makes.
const PAGES: u64 = 2;
/// Its size in bytes.
const SIZE: u64 = PAGES * PAGE_SIZE;

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// What the test server does with a request: runs the test's closure with the request's
/// context and its message.
type Script = Box<dyn FnMut(&mut IpcContext<'_>, &mut [u8]) -> Result<(), HandlerError> + Send>;

struct Scripted(Arc<Mutex<Option<Script>>>);

impl SessionHandler for Scripted {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        _: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        let mut script = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let script = script.as_mut().expect("a script for the request");
        script(cx, message)?;
        Ok(Handled::Replied)
    }

    fn take_reply(
        &mut self,
        _: &mut IpcContext<'_>,
        _: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        None
    }
}

/// A dispatcher over the fixture's address space, with one server on the port `test:`, the
/// guest memory of the code and data, and the log of the address space's translation.
struct World {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    log: MapperLog,
    script: Arc<Mutex<Option<Script>>>,
}

impl World {
    fn new() -> Self {
        let (space, log, _) = fixture::loaded();
        let tables =
            crate::process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
        let dispatcher = Dispatcher::new(tables);
        let script = Arc::new(Mutex::new(None));
        dispatcher
            .add_server(
                "test",
                SERVICE,
                Some(PortName::new("test:").unwrap()),
                Box::new(Scripted(Arc::clone(&script))),
            )
            .unwrap();
        Self {
            dispatcher,
            memory: FlatMemory::new(CODE, usize::try_from(CODE_SIZE).unwrap()),
            env: LiveEnv::new(FixedClock::new(1_000, 1_000), 0),
            recorder: Recorder::new(),
            log,
            script,
        }
    }

    /// Dispatches `id` with `regs`, its message buffer at [`TLS`].
    fn call(&self, id: SvcId, regs: &mut GuestRegs) -> Record {
        regs.tpidrro = TLS;
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 1,
            thread: THREAD,
            env: &self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        self.dispatcher.dispatch(id.number(), regs, &mut caller)
    }

    /// Connects to `test:` and returns the session's handle.
    fn session(&self) -> Handle {
        self.memory.write(NAMES, b"test:\0").unwrap();
        let mut regs = GuestRegs::default();
        regs.x[1] = NAMES;
        self.call(SvcId::ConnectToNamedPort, &mut regs);
        assert_eq!(regs.x[0], 0);
        Handle::from_register(regs.x[1])
    }

    /// Sends `message` on `session` with the server running `script`, and returns the result
    /// in `x0`.
    fn send(&self, session: Handle, message: &RequestBuilder, script: Script) -> u64 {
        *self.script.lock().unwrap() = Some(script);
        let mut buffer = [0; BUFFER_SIZE];
        message.build(&mut buffer).unwrap();
        self.memory.write(TLS, &buffer).unwrap();
        let mut regs = GuestRegs::default();
        regs.x[0] = u64::from(session.raw());
        self.call(SvcId::SendSyncRequest, &mut regs);
        regs.x[0]
    }

    /// The first word of the reply in the message buffer.
    fn reply_word(&self) -> u32 {
        let mut word = [0; 4];
        self.memory.read(TLS, &mut word).unwrap();
        u32::from_le_bytes(word)
    }

    /// Has the server make a shared memory of [`SIZE`] bytes with `permission` and give the
    /// guest a handle to it (the reply's first word); returns the handle and the pages.
    fn shared_memory(&self, permission: SvcPermission) -> (Handle, SharedPages) {
        let session = self.session();
        let made = Arc::new(Mutex::new(None));
        let keep = Arc::clone(&made);
        let result = self.send(
            session,
            &request(),
            Box::new(move |cx, message| {
                let (object, pages) = cx.create_shared_memory(SIZE, permission).unwrap();
                let handle = cx.insert_handle(object).unwrap();
                message[..4].copy_from_slice(&handle.raw().to_le_bytes());
                *keep.lock().unwrap() = Some(pages);
                Ok(())
            }),
        );
        assert_eq!(result, 0);
        let pages = made.lock().unwrap().take().unwrap();
        (Handle::from_raw(self.reply_word()), pages)
    }

    fn block(&self, address: u64) -> (u64, u64, MemoryState, MemoryPermission, MemoryAttribute) {
        self.dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
            let block = space.query_info(address).unwrap();
            (
                block.address(),
                block.pages(),
                block.state(),
                block.permission(),
                block.attribute(),
            )
        })
    }

    fn object(&self, handle: Handle) -> SvcResult<Object> {
        self.dispatcher
            .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
                let id = handles
                    .get(handle)
                    .map_err(|_| KernelError::InvalidHandle)?;
                scheduler
                    .objects()
                    .get(id)
                    .map_err(|_| KernelError::InvalidHandle)
            })
    }
}

fn request() -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, 1).data(&[0x5A; 8])
}

/// The first copy handle of a request, as the guest wrote it.
fn copied_handle(message: &[u8]) -> Handle {
    let word = |index: usize| {
        let bytes: [u8; 4] = message[index * 4..index * 4 + 4].try_into().unwrap();
        u32::from_le_bytes(bytes)
    };
    let special = SpecialHeader::decode(word(2));
    let pid_words = if special.send_pid { 2 } else { 0 };
    Handle::from_raw(word(3 + pid_words))
}

fn code(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

/// Registers that all differ, so a stray write or a read from the wrong register shows.
fn regs() -> GuestRegs {
    let mut regs = GuestRegs::default();
    for (i, x) in (0_u64..).zip(regs.x.iter_mut()) {
        *x = 0x1111_1111_1111_1111_u64.wrapping_mul(i + 1);
    }
    regs
}

/// A 4-byte argument in a register whose upper half holds something else: the call must read
/// only the low four bytes.
fn word(value: u32) -> u64 {
    0xDEAD_BEEF_0000_0000 | u64::from(value)
}

#[test]
fn map_and_unmap_shared_memory_read_their_registers_and_write_only_x0() {
    let world = World::new();
    let (handle, pages) = world.shared_memory(SvcPermission::Read);
    let mut map = regs();
    map.x[0] = u64::from(handle.raw());
    map.x[1] = SHARED;
    map.x[2] = SIZE;
    map.x[3] = word(u32::from(SvcPermission::Read.bits()));
    let before = map.clone();
    drop(world.log.take());
    assert_eq!(
        world.call(SvcId::MapSharedMemory, &mut map).outcome,
        Outcome::Completed
    );
    assert_eq!(map.x[0], 0);
    assert_eq!(map.x[1..], before.x[1..]);
    assert_eq!(
        world.log.take(),
        [MapperCall::MapShared {
            address: SHARED,
            backing: pages.backing(),
            perms: Perms::Read,
        }]
    );
    let mut unmap = regs();
    unmap.x[0] = u64::from(handle.raw());
    unmap.x[1] = SHARED;
    unmap.x[2] = SIZE;
    let before = unmap.clone();
    assert_eq!(
        world.call(SvcId::UnmapSharedMemory, &mut unmap).outcome,
        Outcome::Completed
    );
    assert_eq!(unmap.x[0], 0);
    assert_eq!(unmap.x[1..], before.x[1..]);
    assert_eq!(world.block(SHARED).2, MemoryState::Free);
    // A failure is its result in x0: the range is free now.
    unmap.x[0] = u64::from(handle.raw());
    assert_eq!(
        world.call(SvcId::UnmapSharedMemory, &mut unmap).outcome,
        Outcome::Completed
    );
    assert_eq!(unmap.x[0], code(KernelError::InvalidCurrentMemory));
}

#[test]
fn a_server_makes_shared_memory_the_guest_maps_and_each_sees_the_others_writes() {
    let world = World::new();
    let (handle, pages) = world.shared_memory(SvcPermission::ReadWrite);
    // The guest's handle names the server's shared memory, which the server still holds too.
    let Ok(Object::SharedMemory(memory)) = world.object(handle) else {
        unreachable!("the reply's handle names no shared memory");
    };
    assert_eq!(
        (memory.backing(), memory.user_permission()),
        (pages.backing(), SvcPermission::ReadWrite)
    );
    let mut map = GuestRegs::default();
    map.x[0] = u64::from(handle.raw());
    map.x[1] = SHARED;
    map.x[2] = SIZE;
    map.x[3] = u64::from(SvcPermission::ReadWrite.bits());
    world.call(SvcId::MapSharedMemory, &mut map);
    assert_eq!(map.x[0], 0);
    let guest = world.log.shared_view();
    pages.write(SIZE - 4, &[1, 2, 3, 4]).unwrap();
    let mut seen = [0; 4];
    guest.read(SHARED + SIZE - 4, &mut seen).unwrap();
    assert_eq!(seen, [1, 2, 3, 4]);
    guest.write(SHARED, &[0xEE]).unwrap();
    let mut byte = [0];
    pages.read(0, &mut byte).unwrap();
    assert_eq!(byte, [0xEE]);
}

#[test]
fn a_server_cannot_make_shared_memory_of_no_bytes_or_a_permission_a_mapping_cannot_have() {
    let world = World::new();
    let session = world.session();
    let results = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&results);
    world.send(
        session,
        &request(),
        Box::new(move |cx, _| {
            let mut seen = seen.lock().unwrap();
            for (size, permission) in [
                (0, SvcPermission::Read),
                (u64::MAX, SvcPermission::Read),
                (SIZE, SvcPermission::None),
                (SIZE, SvcPermission::ReadExecute),
            ] {
                seen.push(
                    cx.create_shared_memory(size, permission)
                        .map(|(_, pages)| pages.size()),
                );
            }
            // One byte is a page, and one more is two.
            for size in [1, PAGE_SIZE + 1] {
                seen.push(
                    cx.create_shared_memory(size, SvcPermission::Read)
                        .map(|(_, pages)| pages.size()),
                );
            }
            Ok(())
        }),
    );
    assert_eq!(
        *results.lock().unwrap(),
        [
            Err(KernelError::InvalidSize),
            Err(KernelError::InvalidSize),
            Err(KernelError::InvalidNewMemoryPermission),
            Err(KernelError::InvalidNewMemoryPermission),
            Ok(PAGE_SIZE),
            Ok(2 * PAGE_SIZE),
        ]
    );
}

#[test]
fn a_translation_with_no_backing_to_share_is_the_kernels_failure() {
    let world = World::new();
    let session = world.session();
    world.log.refuse(true);
    let results = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&results);
    world.send(
        session,
        &request(),
        Box::new(move |cx, _| {
            let made = cx.create_shared_memory(SIZE, SvcPermission::Read);
            seen.lock().unwrap().push(made.map(|(id, _)| id));
            Ok(())
        }),
    );
    world.log.refuse(false);
    assert_eq!(*results.lock().unwrap(), [Err(KernelError::InvalidState)]);
    // Nothing was made: the object table holds the session alone.
    let objects = world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().len());
    assert_eq!(objects, 1);
}

/// Lends the data's first page from the guest with `permission` and returns the handle.
fn lend(world: &World, permission: SvcPermission) -> Handle {
    let mut regs = regs();
    regs.x[1] = DATA;
    regs.x[2] = PAGE_SIZE;
    regs.x[3] = word(u32::from(permission.bits()));
    let before = regs.clone();
    assert_eq!(
        world.call(SvcId::CreateTransferMemory, &mut regs).outcome,
        Outcome::Completed
    );
    assert_eq!(regs.x[0], 0);
    assert_eq!(regs.x[2..], before.x[2..]);
    Handle::from_register(regs.x[1])
}

#[test]
fn create_transfer_memory_reads_x1_to_x3_and_returns_its_handle_in_x1() {
    let world = World::new();
    let handle = lend(&world, SvcPermission::Read);
    let Ok(Object::TransferMemory(transfer)) = world.object(handle) else {
        unreachable!("x1 names no transfer memory");
    };
    assert_eq!(
        (
            transfer.address(),
            transfer.size(),
            transfer.owner_permission()
        ),
        (DATA, PAGE_SIZE, SvcPermission::Read)
    );
    assert!(transfer.is_locked());
    assert_eq!(
        world.block(DATA),
        (
            DATA,
            1,
            MemoryState::CodeData,
            MemoryPermission::USER_READ,
            MemoryAttribute::LOCKED
        )
    );
    // A failing call writes its result, and zero for the handle.
    let mut again = regs();
    again.x[1] = DATA;
    again.x[2] = PAGE_SIZE;
    again.x[3] = 0;
    world.call(SvcId::CreateTransferMemory, &mut again);
    assert_eq!(
        (again.x[0], again.x[1]),
        (code(KernelError::InvalidCurrentMemory), 0)
    );
}

#[test]
fn closing_a_transfer_memorys_last_handle_unlocks_its_pages_before_the_lock_goes() {
    let world = World::new();
    let handle = lend(&world, SvcPermission::None);
    assert_eq!(world.block(DATA).4, MemoryAttribute::LOCKED);
    drop(world.log.take());
    let mut close = GuestRegs::default();
    close.x[0] = u64::from(handle.raw());
    world.call(SvcId::CloseHandle, &mut close);
    assert_eq!(close.x[0], 0);
    // The next holder of the lock finds the page as it was before it was lent.
    assert_eq!(
        world.block(DATA),
        (
            DATA,
            fixture::DATA_PAGES,
            MemoryState::CodeData,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE
        )
    );
    assert_eq!(
        world.log.take(),
        [MapperCall::Protect {
            address: DATA,
            size: PAGE_SIZE,
            perms: Perms::ReadWrite,
        }]
    );
    let pending = world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().has_unlocks());
    assert!(!pending);
}

#[test]
fn a_server_reads_and_writes_a_transfer_memory_it_is_sent_within_its_bounds() {
    let world = World::new();
    world.memory.write(DATA + 16, &[7, 8, 9]).unwrap();
    let handle = lend(&world, SvcPermission::None);
    let session = world.session();
    let results = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&results);
    let result = world.send(
        session,
        &request().copy_handle(handle.raw()),
        Box::new(move |cx, message| {
            let mut seen = seen.lock().unwrap();
            let object = cx.resolve_handle(copied_handle(message)).unwrap();
            let transfer: SvcResult<TransferMemory> = cx.transfer_memory(object);
            seen.push(transfer.map(|transfer| (transfer.address(), transfer.size())));
            // The owner may not touch the page, but the server reads it where it lies.
            let mut read = [0; 3];
            cx.read_transfer_memory(object, 16, &mut read).unwrap();
            seen.push(Ok((u64::from(read[0]), u64::from(read[2]))));
            cx.write_transfer_memory(object, PAGE_SIZE - 1, &[0x42])
                .unwrap();
            // One byte past the end, reading or writing, and an offset that wraps.
            let past = cx.read_transfer_memory(object, PAGE_SIZE - 1, &mut [0; 2]);
            seen.push(past.map(|()| (0, 0)));
            let past = cx.write_transfer_memory(object, PAGE_SIZE, &[1]);
            seen.push(past.map(|()| (0, 0)));
            let wraps = cx.read_transfer_memory(object, u64::MAX, &mut [0; 1]);
            seen.push(wraps.map(|()| (0, 0)));
            // A session is not a transfer memory.
            let session = cx.resolve_handle(session).unwrap();
            seen.push(cx.transfer_memory(session).map(|_| (0, 0)));
            Ok(())
        }),
    );
    assert_eq!(result, 0);
    let outside = Err(KernelError::InvalidCurrentMemory);
    assert_eq!(
        *results.lock().unwrap(),
        [
            Ok((DATA, PAGE_SIZE)),
            Ok((7, 9)),
            outside,
            outside,
            outside,
            Err(KernelError::InvalidHandle),
        ]
    );
    let mut last = [0];
    world.memory.read(DATA + PAGE_SIZE - 1, &mut last).unwrap();
    assert_eq!(last, [0x42]);
}
