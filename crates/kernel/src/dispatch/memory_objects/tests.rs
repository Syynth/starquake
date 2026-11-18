//! The shared and transfer memory arms of the dispatcher, and a server's side of both through
//! [`IpcContext`]: which registers each call reads and writes, a shared memory a server makes
//! and the guest maps, a transfer memory the guest sends a server, and the pages a transfer
//! memory's last handle gives back as the kernel lock goes; and a shared memory's pages, which
//! its server takes after every check that can refuse, and which its last reference gives back
//! to the translation as the kernel lock goes.

use std::sync::{Arc, Mutex, PoisonError};

use starquake_cpu::GuestRegs;
use starquake_ipc::{BUFFER_SIZE, CommandType, ReplyBuilder, RequestBuilder, SpecialHeader};
use starquake_memory::{BackingRange, FlatMemory, GuestMemory, PAGE_SIZE, Perms};
use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId};

use super::*;
use crate::dispatch::ipc::create_shared_memory_in;
use crate::dispatch::{Caller, IpcContext, Record};
use crate::id::{ObjectId, SlabId};
use crate::ipc::{Handled, HandlerError, PortName, SessionHandler};
use crate::memory::fixture::{self, CODE, CODE_SIZE, DATA, SHARED};
use crate::memory::testing::{MapperCall, MapperLog, SHARED_BASE, SHARED_CAPACITY};
use crate::memory::{
    MemoryAttribute, MemoryPermission, MemoryState, SharedMemory, SharedPages, SvcPermission,
    TransferMemory,
};
use crate::object::Object;
use crate::process::{Console, ProcessDescription};
use crate::result::{KernelError, SvcError, SvcResult};
use crate::svc::DebugSink;
use crate::svc::handle::remove_handle;

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

    /// Has the server make a shared memory of [`SIZE`] bytes with `permission` and give the
    /// guest a handle to it (the reply's copied handle); returns the handle and the pages.
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
                // The reply names the handle, so the guest keeps it (`dispatch/ipc/placed.rs`).
                ReplyBuilder::cmif(0)
                    .copy_handle(handle.raw())
                    .build(message)
                    .unwrap();
                *keep.lock().unwrap() = Some(pages);
                Ok(())
            }),
        );
        assert_eq!(result, 0);
        let pages = made.lock().unwrap().take().unwrap();
        let mut reply = [0; BUFFER_SIZE];
        self.memory.read(TLS, &mut reply).unwrap();
        (copied_handle(&reply), pages)
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
                // Both faults: the size is checked first.
                (0, SvcPermission::None),
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
            Err(KernelError::InvalidSize),
            Ok(PAGE_SIZE),
            Ok(2 * PAGE_SIZE),
        ]
    );
    // Only the two that were made asked the translation for pages.
    let allocations = world
        .log
        .take()
        .into_iter()
        .filter(|call| matches!(call, MapperCall::AllocateShared { .. }))
        .count();
    assert_eq!(allocations, 2);
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
    // Nothing was made: the object table holds the session alone, besides the process.
    let objects = world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects_besides_processes());
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
    // A transfer memory whose pages were never locked: one `CreateTransferMemory` is still
    // making, which no handle names yet.
    let unlocked = world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| {
            let owner = crate::id::ProcessId::from_parts(0, 1);
            let made = TransferMemory::new(owner, DATA, PAGE_SIZE, SvcPermission::None);
            scheduler
                .objects_mut()
                .create(Object::TransferMemory(made))
                .unwrap()
        });
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
            seen.push(cx.transfer_memory(unlocked).map(|_| (0, 0)));
            let mut byte = [0];
            seen.push(
                cx.read_transfer_memory(unlocked, 0, &mut byte)
                    .map(|()| (0, 0)),
            );
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
            Err(KernelError::InvalidHandle),
            Err(KernelError::InvalidHandle),
        ]
    );
    let mut last = [0];
    world.memory.read(DATA + PAGE_SIZE - 1, &mut last).unwrap();
    assert_eq!(last, [0x42]);
}

// ----- The graphics server's address-space calls, through `IpcContext` -----------------------

#[test]
fn a_server_asks_whether_the_senders_memory_is_gpu_accessible() {
    let world = World::new();
    // The guest marks the data's first page with `svc 0x80`.
    let gpu = u64::from(MemoryAttribute::GPU_ACCESSIBLE.svc());
    let mut regs = GuestRegs::default();
    regs.x[..4].copy_from_slice(&[DATA, PAGE_SIZE, gpu, gpu]);
    world.call(SvcId::SetGpuMemoryAttribute, &mut regs);
    assert_eq!(regs.x[0], 0);
    let answers = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&answers);
    let session = world.session();
    let result = world.send(
        session,
        &request(),
        Box::new(move |cx, _| {
            let mut answers = keep.lock().unwrap();
            for (address, size) in [(DATA, PAGE_SIZE), (DATA, PAGE_SIZE + 1), (DATA - 1, 1)] {
                answers.push(cx.is_gpu_accessible(address, size));
            }
            Ok(())
        }),
    );
    assert_eq!(result, 0);
    assert_eq!(*answers.lock().unwrap(), [true, false, false]);
}

#[test]
fn a_server_locks_the_senders_memory_against_change_and_unlocks_it() {
    let world = World::new();
    let session = world.session();
    // The token the lock returned, which the unlock takes.
    let held = Arc::new(Mutex::new(None));
    let lock = |unlock: bool| -> Script {
        let held = Arc::clone(&held);
        Box::new(move |cx, _| {
            let mut held = held.lock().unwrap();
            let done = if unlock {
                cx.unlock_device_share(held.take().unwrap())
            } else {
                cx.lock_device_share(DATA, PAGE_SIZE).map(|token| {
                    assert_eq!((token.address(), token.size()), (DATA, PAGE_SIZE));
                    *held = Some(token);
                })
            };
            done.map_err(|error| HandlerError(error.to_string()))
        })
    };
    let reprotect = || {
        let mut regs = GuestRegs::default();
        regs.x[..3].copy_from_slice(&[DATA, PAGE_SIZE, u64::from(SvcPermission::Read.bits())]);
        world.call(SvcId::SetMemoryPermission, &mut regs);
        regs.x[0]
    };
    assert_eq!(world.send(session, &request(), lock(false)), 0);
    assert!(world.block(DATA).4.contains(MemoryAttribute::DEVICE_SHARED));
    let spent = held.lock().unwrap().unwrap();
    // While it is locked the guest cannot change it.
    assert_eq!(reprotect(), code(KernelError::InvalidCurrentMemory));
    assert_eq!(world.send(session, &request(), lock(true)), 0);
    assert_eq!(world.block(DATA).4, MemoryAttribute::NONE);
    assert_eq!(reprotect(), 0);
    // The address space's errors reach the server: free memory cannot be locked, and a token
    // whose lock is gone cannot unlock.
    let errors = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&errors);
    let result = world.send(
        session,
        &request(),
        Box::new(move |cx, _| {
            let mut errors = keep.lock().unwrap();
            errors.push(
                cx.lock_device_share(CODE - PAGE_SIZE, PAGE_SIZE)
                    .map(|_| ()),
            );
            errors.push(cx.unlock_device_share(spent));
            Ok(())
        }),
    );
    assert_eq!(result, 0);
    let invalid = Err(KernelError::InvalidCurrentMemory);
    assert_eq!(*errors.lock().unwrap(), [invalid, invalid]);
}

/// Has the server make shared memories of each of `sizes` bytes, readable, and returns what each
/// came to: the size of its pages, or the error.
fn make_shared(world: &World, sizes: &[u64]) -> Vec<SvcResult<u64>> {
    let session = world.session();
    let results = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&results);
    let sizes = sizes.to_vec();
    world.send(
        session,
        &request(),
        Box::new(move |cx, _| {
            let mut seen = seen.lock().unwrap();
            for size in &sizes {
                seen.push(
                    cx.create_shared_memory(*size, SvcPermission::Read)
                        .map(|(_, pages)| pages.size()),
                );
            }
            Ok(())
        }),
    );

    results.lock().unwrap().clone()
}

#[test]
fn a_server_runs_out_of_shared_memory_at_exactly_the_last_page_as_eden_does() {
    let world = World::new();
    // One page more than the translation holds, then all of it, then a page more.
    assert_eq!(
        make_shared(
            &world,
            &[SHARED_CAPACITY + PAGE_SIZE, SHARED_CAPACITY, PAGE_SIZE]
        ),
        [
            Err(KernelError::OutOfMemory),
            Ok(SHARED_CAPACITY),
            Err(KernelError::OutOfMemory),
        ]
    );
    // The refusals asked for pages and got none; the one that fit took the first pages.
    let allocations: Vec<_> = world
        .log
        .take()
        .into_iter()
        .filter(|call| matches!(call, MapperCall::AllocateShared { .. }))
        .collect();
    assert_eq!(
        allocations,
        [MapperCall::AllocateShared {
            backing: BackingRange::new(SHARED_BASE, SHARED_CAPACITY).unwrap()
        }]
    );
}

/// The fixture's address space and a scheduler whose object table holds at most `max_slots`
/// objects, `held` of them taken by events.
fn small_table(max_slots: u32, held: u32) -> (AddressSpace, MapperLog, Scheduler, Vec<ObjectId>) {
    let (space, log, _) = fixture::loaded();
    let mut scheduler = Scheduler::default();
    *scheduler.objects_mut() = crate::object::Objects::with_max_slots(max_slots);
    let taken = (0..held)
        .map(|_| {
            let backing = BackingRange::new(0x1000, PAGE_SIZE).unwrap();
            let memory = SharedMemory::new(backing, SvcPermission::Read).unwrap();
            scheduler
                .objects_mut()
                .create(Object::SharedMemory(memory))
                .unwrap()
        })
        .collect();
    (space, log, scheduler, taken)
}

#[test]
fn a_full_object_table_refuses_before_any_pages_are_taken() {
    let (mut space, log, mut scheduler, taken) = small_table(2, 2);
    // Both faults: no room in the table, and more pages than the translation holds. The table
    // is checked first, and no pages are asked for.
    for size in [SHARED_CAPACITY + PAGE_SIZE, SIZE] {
        assert_eq!(
            create_shared_memory_in(&mut space, &mut scheduler, size, SvcPermission::Read)
                .map(|(_, pages)| pages.size()),
            Err(SvcError::Kernel(KernelError::OutOfResource)),
            "{size:#x}"
        );
    }
    assert_eq!(log.take(), []);
    // One slot free: the next shared memory is made, from the translation's first pages, so
    // the refusals took none.
    let freed = scheduler.objects_mut().release(taken[0]).unwrap();
    assert!(matches!(freed, Some(Object::SharedMemory(_))));
    drop(scheduler.objects_mut().take_frees());
    let (object, pages) =
        create_shared_memory_in(&mut space, &mut scheduler, SIZE, SvcPermission::Read).unwrap();
    assert_eq!(
        pages.backing(),
        BackingRange::new(SHARED_BASE, SIZE).unwrap()
    );
    assert!(matches!(
        scheduler.objects().get(object),
        Ok(Object::SharedMemory(_))
    ));
    // Room, and too many pages: out of memory, and no object made.
    scheduler.objects_mut().release(object).unwrap();
    drop(scheduler.objects_mut().take_frees());
    let before = scheduler.objects().len();
    assert_eq!(
        create_shared_memory_in(
            &mut space,
            &mut scheduler,
            SHARED_CAPACITY,
            SvcPermission::Read
        )
        .map(|(_, pages)| pages.size()),
        Err(SvcError::Kernel(KernelError::OutOfMemory))
    );
    assert_eq!(scheduler.objects().len(), before);
}

#[test]
fn an_object_not_made_after_its_pages_gives_them_back_at_once() {
    // The checked entry point refuses this permission before it takes pages; called past
    // those checks, the object is what refuses, after the pages were taken.
    let (mut space, log, mut scheduler, _) = small_table(4, 0);
    assert_eq!(
        create_shared_memory_in(&mut space, &mut scheduler, SIZE, SvcPermission::None)
            .map(|(_, pages)| pages.size()),
        Err(SvcError::Kernel(KernelError::InvalidNewMemoryPermission))
    );
    let backing = BackingRange::new(SHARED_BASE, SIZE).unwrap();
    assert_eq!(
        log.take(),
        [
            MapperCall::AllocateShared { backing },
            MapperCall::ReleaseShared { backing },
        ]
    );
    assert!(scheduler.objects().is_empty());
    assert!(!scheduler.objects().has_frees());
}

#[test]
fn giving_back_waits_for_the_last_reference_and_goes_after_the_unmap() {
    let world = World::new();
    let (handle, pages) = world.shared_memory(SvcPermission::ReadWrite);
    let object = world
        .dispatcher
        .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap());
    let mut map = GuestRegs::default();
    map.x[0] = u64::from(handle.raw());
    map.x[1] = SHARED;
    map.x[2] = SIZE;
    map.x[3] = u64::from(SvcPermission::ReadWrite.bits());
    world.call(SvcId::MapSharedMemory, &mut map);
    assert_eq!(map.x[0], 0);
    drop(world.log.take());
    let refs = || {
        world
            .dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().refs(object))
    };
    // The guest closes its handle, and the server lets go: the mapping's reference is left.
    let mut close = GuestRegs::default();
    close.x[0] = u64::from(handle.raw());
    world.call(SvcId::CloseHandle, &mut close);
    assert_eq!(close.x[0], 0);
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.close_object(object))
        .unwrap();
    assert_eq!(refs(), 1);
    assert_eq!(world.log.take(), []);
    // The guest cannot unmap through the closed handle; the kernel unmaps the mapping, as
    // `UnmapSharedMemory` does, and drops its reference: the pages go back after the unmap,
    // before the lock is let go.
    world
        .dispatcher
        .with::<(&mut AddressSpace, &mut Scheduler), _>(|(space, scheduler)| {
            space.unmap_shared(SHARED, SIZE, object)?;
            scheduler.close_object(object)
        })
        .unwrap();
    assert_eq!(refs(), 0);
    assert_eq!(
        world.log.take(),
        [
            MapperCall::Unmap {
                address: SHARED,
                size: SIZE
            },
            MapperCall::ReleaseShared {
                backing: pages.backing()
            },
        ]
    );
    let pending = world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().has_frees());
    assert!(!pending);
}

#[test]
fn closing_the_last_handle_after_the_unmap_gives_the_pages_back() {
    let world = World::new();
    let (handle, pages) = world.shared_memory(SvcPermission::Read);
    let object = world
        .dispatcher
        .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap());
    let mut map = GuestRegs::default();
    map.x[0] = u64::from(handle.raw());
    map.x[1] = SHARED;
    map.x[2] = SIZE;
    map.x[3] = u64::from(SvcPermission::Read.bits());
    world.call(SvcId::MapSharedMemory, &mut map);
    assert_eq!(map.x[0], 0);
    // The server lets go; the guest's handle and its mapping are left.
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.close_object(object))
        .unwrap();
    drop(world.log.take());
    // Unmapping leaves the handle's reference: nothing given back.
    let mut unmap = GuestRegs::default();
    unmap.x[0] = u64::from(handle.raw());
    unmap.x[1] = SHARED;
    unmap.x[2] = SIZE;
    world.call(SvcId::UnmapSharedMemory, &mut unmap);
    assert_eq!(unmap.x[0], 0);
    assert_eq!(
        world.log.take(),
        [MapperCall::Unmap {
            address: SHARED,
            size: SIZE
        }]
    );
    // Closing the handle drops the last reference: given back as the call ends.
    let mut close = GuestRegs::default();
    close.x[0] = u64::from(handle.raw());
    world.call(SvcId::CloseHandle, &mut close);
    assert_eq!(close.x[0], 0);
    assert_eq!(
        world.log.take(),
        [MapperCall::ReleaseShared {
            backing: pages.backing()
        }]
    );
}

#[test]
fn a_refused_give_back_of_an_object_not_made_returns_the_objects_error() {
    // As `an_object_not_made_after_its_pages_gives_them_back_at_once`, but the translation
    // refuses the give-back too: the object's error is still what the caller gets.
    let (mut space, log, mut scheduler, _) = small_table(4, 0);
    log.refuse_calls(&[1]);
    assert_eq!(
        create_shared_memory_in(&mut space, &mut scheduler, SIZE, SvcPermission::None)
            .map(|(_, pages)| pages.size()),
        Err(SvcError::Kernel(KernelError::InvalidNewMemoryPermission))
    );
    assert_eq!(
        log.take(),
        [MapperCall::AllocateShared {
            backing: BackingRange::new(SHARED_BASE, SIZE).unwrap()
        }]
    );
    assert!(scheduler.objects().is_empty());
    assert!(!scheduler.objects().has_frees());
}

/// Has the server make a shared memory, and lets go of it, so the guest's handle holds its
/// only reference (it is not mapped); then lends the data's first page, so a second handle
/// holds a locked transfer memory's only reference. Returns the two handles, shared memory's
/// first, and the shared memory's pages, with the log emptied.
fn shared_then_lent(world: &World) -> (Handle, Handle, SharedPages) {
    let (handle, pages) = world.shared_memory(SvcPermission::Read);
    let object = world
        .dispatcher
        .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap());
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.close_object(object))
        .unwrap();
    let lent = lend(world, SvcPermission::None);
    assert_eq!(world.block(DATA).4, MemoryAttribute::LOCKED);
    drop(world.log.take());
    (handle, lent, pages)
}

/// Whether the object table still holds released transfer memories to unlock, and released
/// shared memories to give back.
fn pending(world: &World) -> (bool, bool) {
    world.dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
        (
            scheduler.objects().has_unlocks(),
            scheduler.objects().has_frees(),
        )
    })
}

#[test]
fn exiting_the_process_unlocks_lent_pages_before_it_gives_back_shared_ones() {
    let world = World::new();
    let (_, _, pages) = shared_then_lent(&world);
    // Teardown closes the shared memory's handle before the transfer memory's (lowest index
    // first), so the order below is the drain's, not the closing's.
    let mut exit = GuestRegs::default();
    assert_eq!(
        world.call(SvcId::ExitProcess, &mut exit).outcome,
        Outcome::ProcessExited
    );
    assert_eq!(
        world.log.take(),
        [
            MapperCall::Protect {
                address: DATA,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite,
            },
            MapperCall::ReleaseShared {
                backing: pages.backing()
            },
        ]
    );
    assert_eq!(pending(&world), (false, false));
}

#[test]
fn exiting_the_process_releases_device_locks_and_keeps_the_unlock_and_give_back_order() {
    // A lent page, a shared memory to give back, and two device-share locks on the page the
    // port name was written to (`dispatch/device.rs`, "When the process ends"): the releases
    // make no translation call, so the calls are still the unlock, then the give-back.
    let world = World::new();
    let (_, _, pages) = shared_then_lent(&world);
    let (_, _, state, permission, _) = world.block(NAMES);
    for _ in 0..2 {
        world
            .dispatcher
            .lock_device_share(NAMES, PAGE_SIZE)
            .unwrap();
    }
    assert_eq!(world.block(NAMES).4, MemoryAttribute::DEVICE_SHARED);
    drop(world.log.take());
    let mut exit = GuestRegs::default();
    assert_eq!(
        world.call(SvcId::ExitProcess, &mut exit).outcome,
        Outcome::ProcessExited
    );
    assert_eq!(
        world.log.take(),
        [
            MapperCall::Protect {
                address: DATA,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite,
            },
            MapperCall::ReleaseShared {
                backing: pages.backing()
            },
        ]
    );
    assert_eq!(pending(&world), (false, false));
    // The page is as it was before the locks: its block (merged with the unlocked page, now
    // that neither is locked) has no attribute.
    let (_, _, after_state, after_permission, attribute) = world.block(NAMES);
    assert_eq!(
        (after_state, after_permission, attribute),
        (state, permission, MemoryAttribute::NONE)
    );
    let locks = world
        .dispatcher
        .with::<(&AddressSpace,), _>(|(space,)| space.device_share_locks().count());
    assert_eq!(locks, 0);
}

#[test]
fn a_refused_unlock_still_gives_back_shared_memory_under_the_same_lock() {
    let world = World::new();
    let (shared, lent, pages) = shared_then_lent(&world);
    // Both last references go while the lock is held once, as in one call (a dispatch such as
    // `ExitProcess` takes the lock more than once, and a later hold would drain what an
    // earlier one left). The unlock's `Protect` is the first translation call.
    world.log.refuse_calls(&[0]);
    world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            remove_handle(handles, scheduler, shared)?;
            remove_handle(handles, scheduler, lent)
        })
        .unwrap();
    assert_eq!(
        world.log.take(),
        [MapperCall::ReleaseShared {
            backing: pages.backing()
        }]
    );
    assert_eq!(pending(&world), (false, false));
}

#[test]
fn a_refused_give_back_still_completes_the_call_and_leaves_none_pending() {
    let world = World::new();
    let (handle, _) = world.shared_memory(SvcPermission::Read);
    let object = world
        .dispatcher
        .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap());
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.close_object(object))
        .unwrap();
    drop(world.log.take());
    // Closing the handle drops the last reference; the translation refuses the give-back.
    world.log.refuse_calls(&[0]);
    let mut close = GuestRegs::default();
    close.x[0] = u64::from(handle.raw());
    world.call(SvcId::CloseHandle, &mut close);
    assert_eq!(close.x[0], 0);
    assert_eq!(world.log.take(), []);
    assert_eq!(pending(&world), (false, false));
    let refs = world
        .dispatcher
        .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().refs(object));
    assert_eq!(refs, 0);
}
