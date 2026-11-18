//! What a server's restore asks of the kernel it is restored beside: the [`RestoredKernel`]
//! trait, through which a family's check ([`Server::with_family_objects`]) finds out whether
//! each kernel object id its snapshot names is one the restored kernel actually holds, and of
//! the kind the family keeps there (the house rule that an id read from possibly corrupt data
//! is validated against what was actually allocated).
//!
//! The kernel's [`Dispatcher`] implements it, over the restored tables; `FakeKernel` (in
//! `framework::fake`, behind the `test-support` feature) implements it over its own, with the
//! difference listed in its module docs.
//!
//! The check asks the kernel's tables as they are when the server is restored, so a run
//! restores a server beside the kernel whose objects its snapshot names, after those objects
//! are in place. The kernel's own object snapshot (`Dispatcher::restore_objects`) checks
//! nothing a service keeps: this is where a service's ids are checked. Which server made an
//! event is the IPC table's note, which has no snapshot of its own yet (the kernel's
//! `dispatch/snapshot.rs`), so today the check passes only beside the tables the server ran
//! on.
//!
//! The [`Dispatcher`]'s answers take its table lock (see the impl), which is not re-entrant, so
//! a run restores its servers on a thread that does not hold that lock:
//! [`Server::restore_beside`] with the dispatcher as the kernel must not be called from inside a
//! [`Dispatcher::with`] closure or a [`Dispatcher::inspect_handlers`] closure (both run with the
//! lock held; that is where a whole-run restore would reach the installed servers). Such a call
//! deadlocks (or panics: std's `Mutex` leaves which unspecified) on its first check of an id.
//! Called from another thread while the lock is held, it waits for the lock like a dispatch.
//!
//! [`Server::with_family_objects`]: super::Server::with_family_objects
//! [`Server::restore_beside`]: super::Server::restore_beside

use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SharedMemory;
use starquake_kernel::object::Object;
use starquake_kernel::tables::Scheduler;
use starquake_trace::ServiceId;

/// The kernel a server is restored beside, as a family's restore check sees it: the objects it
/// holds, by id. Every call is a question about the tables as they are now; none changes them.
///
/// An implementation may take a lock to answer: the [`Dispatcher`]'s takes its table lock on
/// every call (the impl says how often), so it must not be asked on a thread that holds that
/// lock (module docs).
pub trait RestoredKernel {
    /// Whether `event` is a live readable end of an event that the server known as `service`
    /// made: what [`Kernel::create_event`](super::Kernel::create_event) returns, and the second
    /// id [`Kernel::create_hooked_event`](super::Kernel::create_hooked_event) returns (the
    /// kernel notes both as the server's and does not tell them apart). False for any other
    /// id: one that names nothing, the writable end of an event, an object of another kind, an
    /// event another server or the guest made, and any id for a service with no server.
    fn is_service_readable_event(&self, service: ServiceId, event: ObjectId) -> bool;

    /// The live shared memory `object` names; `None` when it names nothing live, or an object
    /// of another kind. Which server made it is not asked: the kernel does not note it (module
    /// docs of each family that keeps one say what that leaves unchecked).
    fn shared_memory(&self, object: ObjectId) -> Option<SharedMemory>;
}

/// Each call takes the dispatcher's table lock through [`Dispatcher::with`] and releases it
/// before returning. The lock is not re-entrant, so neither call may be made on a thread that
/// holds it: not from inside a [`Dispatcher::with`] or [`Dispatcher::inspect_handlers`] closure
/// (it deadlocks or panics). Neither holds the lock while it calls anything outside the kernel.
impl RestoredKernel for Dispatcher {
    /// A live [`Object::ReadableEvent`] that [`Dispatcher::is_service_event`] counts as the
    /// server's. Takes the table lock twice, one after the other: once here for the object's
    /// kind, then once in [`Dispatcher::is_service_event`] (only when the kind matches).
    fn is_service_readable_event(&self, service: ServiceId, event: ObjectId) -> bool {
        let readable = self.with::<(&Scheduler,), _>(|(scheduler,)| {
            matches!(scheduler.objects().get(event), Ok(Object::ReadableEvent(_)))
        });
        readable && self.is_service_event(service, event)
    }

    /// The object table's row for `object`, if it is an [`Object::SharedMemory`]. Takes the
    /// table lock once.
    fn shared_memory(&self, object: ObjectId) -> Option<SharedMemory> {
        self.with::<(&Scheduler,), _>(|(scheduler,)| match scheduler.objects().get(object) {
            Ok(Object::SharedMemory(memory)) => Some(memory),
            _ => None,
        })
    }
}
