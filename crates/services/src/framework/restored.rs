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
//! [`Server::with_family_objects`]: super::Server::with_family_objects

use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::SharedMemory;
use starquake_kernel::object::Object;
use starquake_kernel::tables::Scheduler;
use starquake_trace::ServiceId;

/// The kernel a server is restored beside, as a family's restore check sees it: the objects it
/// holds, by id. Every call is a question about the tables as they are now; none changes them.
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

impl RestoredKernel for Dispatcher {
    /// A live [`Object::ReadableEvent`] that [`Dispatcher::is_service_event`] counts as the
    /// server's.
    fn is_service_readable_event(&self, service: ServiceId, event: ObjectId) -> bool {
        let readable = self.with::<(&Scheduler,), _>(|(scheduler,)| {
            matches!(scheduler.objects().get(event), Ok(Object::ReadableEvent(_)))
        });
        readable && self.is_service_event(service, event)
    }

    /// The object table's row for `object`, if it is an [`Object::SharedMemory`].
    fn shared_memory(&self, object: ObjectId) -> Option<SharedMemory> {
        self.with::<(&Scheduler,), _>(|(scheduler,)| match scheduler.objects().get(object) {
            Ok(Object::SharedMemory(memory)) => Some(memory),
            _ => None,
        })
    }
}
