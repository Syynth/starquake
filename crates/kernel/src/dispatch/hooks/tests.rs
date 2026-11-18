//! Hooks on service-created events: a server makes a hooked event and gives the guest its
//! writable end, and the guest's `SignalEvent` runs the hook once the kernel lock is released.

use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use starquake_cpu::GuestRegs;
use starquake_ipc::{BUFFER_SIZE, CommandType, ReplyBuilder, RequestBuilder};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId};

use super::*;
use crate::dispatch::{Caller, IpcContext, Record};
use crate::handle::{Handle, HandleTable};
use crate::ipc::{Handled, HandlerError, PortName, SessionHandler};
use crate::memory::fixture::{self, CODE, CODE_SIZE, DATA};
use crate::object::Object;
use crate::process::{Console, ProcessDescription};
use crate::result::KernelError;
use crate::slab::SlabId;
use crate::svc::{DebugSink, SvcId};
use crate::tables::Scheduler;

/// The guest thread of these tests.
const THREAD: ThreadId = ThreadId(0xB2);
/// Its message buffer: the data's second page.
const TLS: u64 = DATA + PAGE_SIZE;
/// Where the port name goes: the data's last page.
const NAMES: u64 = DATA + 2 * PAGE_SIZE;
/// The test server's service number.
const SERVICE: ServiceId = ServiceId(0x52);
/// The longest a test waits for a dispatch on another thread: a dispatch that would deadlock
/// fails the test instead of hanging the gate.
const TIMEOUT: Duration = Duration::from_secs(10);

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// What the test server does with a request.
type Script = Box<dyn FnMut(&mut IpcContext<'_>, &mut [u8]) + Send>;

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
        (script.as_mut().expect("a script for the request"))(cx, message);
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

/// A hook that counts what it is told, and runs a probe as it is told.
#[derive(Default)]
struct Counting {
    signalled: Mutex<Vec<ObjectId>>,
}

impl Counting {
    fn signalled(&self) -> Vec<ObjectId> {
        self.signalled
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl EventHook for Counting {
    fn signalled(&self, event: ObjectId) {
        self.signalled
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

/// A dispatcher over the fixture's process with one server on `hook:`.
struct World {
    dispatcher: Arc<Dispatcher>,
    memory: Arc<FlatMemory>,
    script: Arc<Mutex<Option<Script>>>,
}

impl World {
    fn new() -> Self {
        let (space, _log, _) = fixture::loaded();
        let tables =
            crate::process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
        let dispatcher = Dispatcher::new(tables);
        let script = Arc::new(Mutex::new(None));
        dispatcher
            .add_server(
                "hook",
                SERVICE,
                Some(PortName::new("hook:").unwrap()),
                Box::new(Scripted(Arc::clone(&script))),
            )
            .unwrap();
        Self {
            dispatcher: Arc::new(dispatcher),
            memory: Arc::new(FlatMemory::new(CODE, usize::try_from(CODE_SIZE).unwrap())),
            script,
        }
    }

    /// Dispatches `id` with `regs` on `dispatcher`, its message buffer at [`TLS`].
    fn call_on(
        dispatcher: &Dispatcher,
        memory: &FlatMemory,
        id: SvcId,
        regs: &mut GuestRegs,
    ) -> Record {
        regs.tpidrro = TLS;
        let env = LiveEnv::new(FixedClock::new(1_000, 1_000), 0);
        let recorder = Recorder::new();
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 2,
            thread: THREAD,
            env: &env,
            memory,
            debug: &mut debug,
            recorder: &recorder,
        };
        dispatcher.dispatch(id.number(), regs, &mut caller)
    }

    fn call(&self, id: SvcId, regs: &mut GuestRegs) -> Record {
        Self::call_on(&self.dispatcher, &self.memory, id, regs)
    }

    /// Makes a call that takes one handle in `x0` and returns only a result; returns the
    /// result.
    fn call_with_handle(&self, id: SvcId, handle: Handle) -> u64 {
        let mut regs = GuestRegs::default();
        regs.x[0] = u64::from(handle.raw());
        self.call(id, &mut regs);
        regs.x[0]
    }

    /// Connects to `hook:` and sends one request with the server running `script`. Returns
    /// the result of `SendSyncRequest`.
    fn request(&self, script: Script) -> u64 {
        self.memory.write(NAMES, b"hook:\0").unwrap();
        let mut regs = GuestRegs::default();
        regs.x[1] = NAMES;
        self.call(SvcId::ConnectToNamedPort, &mut regs);
        assert_eq!(regs.x[0], 0);
        let session = Handle::from_register(regs.x[1]);
        *self.script.lock().unwrap() = Some(script);
        let mut buffer = [0; BUFFER_SIZE];
        RequestBuilder::cmif(CommandType::Request, 1)
            .build(&mut buffer)
            .unwrap();
        self.memory.write(TLS, &buffer).unwrap();
        self.call_with_handle(SvcId::SendSyncRequest, session)
    }

    /// Has the server make a hooked event with `hook` and give the guest a handle to its
    /// writable end. Returns that handle and the two ends.
    fn hooked_event(&self, hook: Arc<dyn EventHook>) -> (Handle, ObjectId, ObjectId) {
        let made = Arc::new(Mutex::new(None));
        let keep = Arc::clone(&made);
        let result = self.request(Box::new(move |cx, message| {
            let (writable, readable) = cx.create_hooked_event(Arc::clone(&hook)).unwrap();
            let handle = cx.insert_handle(writable).unwrap();
            // The reply names the handle, so the guest keeps it (`dispatch/ipc/placed.rs`).
            ReplyBuilder::cmif(0)
                .copy_handle(handle.raw())
                .build(message)
                .unwrap();
            *keep.lock().unwrap() = Some((handle, writable, readable));
        }));
        assert_eq!(result, 0);
        made.lock().unwrap().take().unwrap()
    }

    /// Whether the readable end `readable` is signalled: `None` if it is not a readable end.
    fn is_signalled(&self, readable: ObjectId) -> Option<bool> {
        self.dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
            match scheduler.objects().get(readable) {
                Ok(Object::ReadableEvent(state)) => Some(state.is_signalled()),
                _ => None,
            }
        })
    }
}

#[test]
fn a_hook_runs_after_the_guest_signals_its_event() {
    let world = World::new();
    let hook = Arc::new(Counting::default());
    let (handle, writable, readable) = world.hooked_event(Arc::<Counting>::clone(&hook));
    assert_ne!(writable, readable);
    assert_eq!(hook.signalled(), []);
    // The guest signals: the event is signalled and the hook told, with the writable end.
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, handle), 0);
    assert_eq!(world.is_signalled(readable), Some(true));
    assert_eq!(hook.signalled(), [writable]);
    // Signalling an event that is already signalled changes nothing in the kernel, and still
    // runs the hook: the guest asked again.
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, handle), 0);
    assert_eq!(world.is_signalled(readable), Some(true));
    assert_eq!(hook.signalled(), [writable, writable]);
}

#[test]
fn each_hook_hears_only_its_own_event() {
    let world = World::new();
    let first = Arc::new(Counting::default());
    let second = Arc::new(Counting::default());
    let (first_handle, first_event, _) = world.hooked_event(Arc::<Counting>::clone(&first));
    let (second_handle, second_event, _) = world.hooked_event(Arc::<Counting>::clone(&second));
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, second_handle), 0);
    assert_eq!(
        (first.signalled(), second.signalled()),
        (vec![], vec![second_event])
    );
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, first_handle), 0);
    assert_eq!(first.signalled(), [first_event]);
    assert_eq!(second.signalled(), [second_event]);
}

#[test]
fn a_hook_runs_only_for_a_successful_guest_signal() {
    let world = World::new();
    let hook = Arc::new(Counting::default());
    let (handle, writable, readable) = world.hooked_event(Arc::<Counting>::clone(&hook));
    // A signal that fails: a handle that names nothing.
    assert_eq!(
        world.call_with_handle(SvcId::SignalEvent, Handle::from_raw(0x7fff_8001)),
        u64::from(KernelError::InvalidHandle.code().raw())
    );
    // Clearing the writable end.
    assert_eq!(world.call_with_handle(SvcId::ClearEvent, handle), 0);
    // The server's own signal, in a later request.
    let result = world.request(Box::new(move |cx, _| cx.signal_event(readable).unwrap()));
    assert_eq!(result, 0);
    assert_eq!(world.is_signalled(readable), Some(true));
    // A host thread's signal.
    let env = LiveEnv::new(FixedClock::new(1_000, 1_000), 0);
    world
        .dispatcher
        .signal_event(writable, &env, &Recorder::new())
        .unwrap();
    assert_eq!(hook.signalled(), []);
    // An event the guest made itself has no hook: signalling it runs nothing.
    let mut regs = GuestRegs::default();
    world.call(SvcId::CreateEvent, &mut regs);
    assert_eq!(regs.x[0], 0);
    let own = Handle::from_register(regs.x[1]);
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, own), 0);
    assert_eq!(hook.signalled(), []);
}

/// A hook that takes the kernel lock while it runs, as a hook must never do: it would
/// deadlock if the dispatch still held the lock. It reports what it found.
struct TakesTheLock {
    dispatcher: Mutex<Option<Arc<Dispatcher>>>,
    found: Mutex<mpsc::Sender<bool>>,
}

impl EventHook for TakesTheLock {
    fn signalled(&self, event: ObjectId) {
        let dispatcher = self
            .dispatcher
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .expect("the dispatcher");
        // The readable end is already signalled when the hook runs.
        let signalled = dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
            match scheduler.objects().get(event) {
                Ok(Object::Event(event)) => matches!(
                    scheduler.objects().get(event.readable()),
                    Ok(Object::ReadableEvent(state)) if state.is_signalled()
                ),
                _ => false,
            }
        });
        let found = self.found.lock().unwrap_or_else(PoisonError::into_inner);
        found.send(signalled).unwrap();
    }
}

#[test]
fn a_hook_runs_without_the_kernel_lock() {
    let world = World::new();
    let (found_tx, found_rx) = mpsc::channel();
    let hook = Arc::new(TakesTheLock {
        dispatcher: Mutex::new(Some(Arc::clone(&world.dispatcher))),
        found: Mutex::new(found_tx),
    });
    let (handle, _, _) = world.hooked_event(Arc::<TakesTheLock>::clone(&hook));
    // The guest's call runs on a thread of its own, so a deadlock fails the test at the
    // timeout instead of hanging it.
    let dispatcher = Arc::clone(&world.dispatcher);
    let memory = Arc::clone(&world.memory);
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut regs = GuestRegs::default();
        regs.x[0] = u64::from(handle.raw());
        World::call_on(&dispatcher, &memory, SvcId::SignalEvent, &mut regs);
        done_tx.send(regs.x[0]).unwrap();
    });
    assert_eq!(found_rx.recv_timeout(TIMEOUT), Ok(true));
    assert_eq!(done_rx.recv_timeout(TIMEOUT), Ok(0));
    // The cycle through the dispatcher is broken so both can go.
    *hook.dispatcher.lock().unwrap() = None;
}

#[test]
fn hooks_of_events_that_are_gone_are_dropped_when_another_is_attached() {
    let hooks = Hooks::default();
    let mut objects = crate::object::Objects::new();
    let mut event = || {
        objects
            .create(Object::ReadableEvent(crate::sync::ReadableEvent::default()))
            .unwrap()
    };
    let (first, second, third) = (event(), event(), event());
    let hook = || -> Arc<dyn EventHook> { Arc::new(Counting::default()) };
    hooks.attach(&objects, first, hook());
    hooks.attach(&objects, second, hook());
    assert_eq!(hooks.len(), 2);
    // The first goes; the next attach drops its hook and keeps the second's.
    assert!(objects.release(first).unwrap().is_some());
    hooks.attach(&objects, third, hook());
    assert_eq!(hooks.len(), 2);
    assert!(hooks.get(first).is_none());
    assert!(hooks.get(second).is_some() && hooks.get(third).is_some());
}

#[test]
fn the_hooked_events_ends_are_a_writable_end_for_the_guest_and_its_readable_end() {
    let world = World::new();
    let (handle, writable, readable) = world.hooked_event(Arc::new(Counting::default()));
    world
        .dispatcher
        .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
            assert_eq!(handles.get(handle), Ok(writable));
            assert!(matches!(
                scheduler.objects().get(writable),
                Ok(Object::Event(event)) if event.readable() == readable
            ));
            // The server's reference and the guest's handle.
            assert_eq!(scheduler.objects().refs(writable), 2);
            assert_eq!(scheduler.objects().refs(readable), 1);
        });
}

/// The object a handle in the guest's table names.
fn object_of(world: &World, handle: Handle) -> ObjectId {
    world
        .dispatcher
        .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap())
}

#[test]
fn a_hook_reattached_after_a_restore_runs_when_the_guest_signals() {
    let world = World::new();
    let old = Arc::new(Counting::default());
    let (handle, writable, _) = world.hooked_event(Arc::<Counting>::clone(&old));
    // What a restored kernel holds: the event, and no hook (hooks are code, not snapshotted).
    world.dispatcher.hooks.clear();
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, handle), 0);
    assert_eq!(old.signalled(), []);
    // The service re-attaches its hook, and the guest's next signal runs it.
    let new = Arc::new(Counting::default());
    let hook: Arc<dyn EventHook> = Arc::<Counting>::clone(&new);
    assert_eq!(
        world
            .dispatcher
            .reattach_event_hook(SERVICE, writable, hook),
        Ok(())
    );
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, handle), 0);
    assert_eq!(new.signalled(), [writable]);
    assert_eq!(old.signalled(), []);
}

#[test]
fn a_hook_is_reattached_only_to_a_writable_end_its_service_made() {
    let world = World::new();
    let (handle, writable, readable) = world.hooked_event(Arc::new(Counting::default()));
    world.dispatcher.hooks.clear();
    // An event the guest made itself: its writable end.
    let mut regs = GuestRegs::default();
    world.call(SvcId::CreateEvent, &mut regs);
    assert_eq!(regs.x[0], 0);
    let guests = object_of(&world, Handle::from_register(regs.x[1]));
    let hook = Arc::new(Counting::default());
    let refused = Err(KernelError::InvalidHandle);
    let other_service = ServiceId(SERVICE.0 + 1);
    let never_made = ObjectId::from_parts(0x7FFF, 9);
    for (service, event) in [
        // The readable end, which the guest cannot signal.
        (SERVICE, readable),
        // The right end, named for another service.
        (other_service, writable),
        // The guest's own event.
        (SERVICE, guests),
        // An id that names nothing.
        (SERVICE, never_made),
    ] {
        let attached: Arc<dyn EventHook> = Arc::<Counting>::clone(&hook);
        assert_eq!(
            world
                .dispatcher
                .reattach_event_hook(service, event, attached),
            refused,
            "{service:?} {event:?}"
        );
    }
    // Nothing was attached: the guest's signal runs no hook.
    assert_eq!(world.dispatcher.hooks.len(), 0);
    assert_eq!(world.call_with_handle(SvcId::SignalEvent, handle), 0);
    assert_eq!(hook.signalled(), []);
}

/// Has the server make a plain event ([`IpcContext::create_event`]); returns its readable end.
fn plain_event(world: &World) -> ObjectId {
    let made = Arc::new(Mutex::new(None));
    let keep = Arc::clone(&made);
    let result = world.request(Box::new(move |cx, _| {
        *keep.lock().unwrap() = Some(cx.create_event().unwrap());
    }));
    assert_eq!(result, 0);
    made.lock().unwrap().take().unwrap()
}

#[test]
fn a_service_event_is_a_live_end_of_one_its_server_made_and_nothing_else() {
    let world = World::new();
    let (_, writable, readable) = world.hooked_event(Arc::new(Counting::default()));
    let plain = plain_event(&world);
    // Either end of the hooked event, and the plain event's readable end.
    for event in [writable, readable, plain] {
        assert!(
            world.dispatcher.is_service_event(SERVICE, event),
            "{event:?}"
        );
    }
    let mut regs = GuestRegs::default();
    world.call(SvcId::CreateEvent, &mut regs);
    assert_eq!(regs.x[0], 0);
    let guests = object_of(&world, Handle::from_register(regs.x[1]));
    let other_service = ServiceId(SERVICE.0 + 1);
    for (service, event) in [
        // The right event, named for another service (which has no server).
        (other_service, writable),
        // The guest's own event, either end.
        (SERVICE, guests),
        // An id that names nothing.
        (SERVICE, ObjectId::from_parts(0x7FFF, 9)),
    ] {
        assert!(
            !world.dispatcher.is_service_event(service, event),
            "{service:?} {event:?}"
        );
    }
}

#[test]
fn the_dispatcher_finds_a_live_transfer_memory_outside_any_request() {
    let world = World::new();
    let mut regs = GuestRegs::default();
    regs.x[1] = DATA;
    regs.x[2] = PAGE_SIZE;
    regs.x[3] = u64::from(crate::memory::SvcPermission::Read.bits());
    world.call(SvcId::CreateTransferMemory, &mut regs);
    assert_eq!(regs.x[0], 0);
    let handle = Handle::from_register(regs.x[1]);
    let lent = object_of(&world, handle);
    let found = world.dispatcher.transfer_memory(lent).unwrap();
    assert_eq!((found.address(), found.size()), (DATA, PAGE_SIZE));
    // An event is not one, nor an id that names nothing.
    let (_, writable, _) = world.hooked_event(Arc::new(Counting::default()));
    for object in [writable, ObjectId::from_parts(0x7FFF, 9)] {
        assert_eq!(
            world.dispatcher.transfer_memory(object),
            Err(KernelError::InvalidHandle)
        );
    }
    // Closed, it is gone.
    assert_eq!(world.call_with_handle(SvcId::CloseHandle, handle), 0);
    assert_eq!(
        world.dispatcher.transfer_memory(lent),
        Err(KernelError::InvalidHandle)
    );
}
