//! The dispatcher's side of [`EventHook`]: which hook each hooked event has, and running a hook
//! after a guest's `SignalEvent` (`crate::ipc::hook` says exactly when).
//!
//! The hooks are kept outside the tables, as servers' handlers are: they are code, not state a
//! snapshot can hold, and they run without the kernel lock. They have a lock of their own,
//! which is only ever taken alone or inside the kernel lock (never the other way round), and
//! is held only to read or change the map, never while a hook runs.
//!
//! A service's restore step re-attaches its hooks here
//! ([`Dispatcher::reattach_event_hook`]), and checks the other kernel objects its snapshot names
//! against what the kernel holds: the events its server made
//! ([`Dispatcher::is_service_event`]) and the transfer memories it was lent
//! ([`Dispatcher::transfer_memory`]).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use starquake_trace::ServiceId;

use super::Dispatcher;
use crate::id::ObjectId;
use crate::ipc::EventHook;
use crate::memory::TransferMemory;
use crate::object::{Object, Objects};
use crate::result::{KernelError, SvcResult};
use crate::tables::Scheduler;

#[cfg(test)]
mod tests;

/// The hook of each hooked event, by the event's writable end.
#[derive(Default)]
pub(super) struct Hooks {
    hooks: Mutex<BTreeMap<ObjectId, Arc<dyn EventHook>>>,
}

impl Hooks {
    /// Attaches `hook` to `event`, a writable end just made in `objects`. Hooks of events whose
    /// writable end is gone are dropped first, so the map holds at most one hook per live object
    /// (an id is never reused: its generation changes).
    pub(super) fn attach(&self, objects: &Objects, event: ObjectId, hook: Arc<dyn EventHook>) {
        let mut hooks = self.hooks.lock().unwrap_or_else(PoisonError::into_inner);
        hooks.retain(|event, _| objects.get(*event).is_ok());
        hooks.insert(event, hook);
    }

    /// The events with a hook, in id order: what a snapshot saves of the hooks
    /// (`dispatch/snapshot.rs`).
    pub(super) fn hooked(&self) -> Vec<ObjectId> {
        let hooks = self.hooks.lock().unwrap_or_else(PoisonError::into_inner);
        hooks.keys().copied().collect()
    }

    /// The hook attached to `event`, if any.
    fn get(&self, event: ObjectId) -> Option<Arc<dyn EventHook>> {
        let hooks = self.hooks.lock().unwrap_or_else(PoisonError::into_inner);
        hooks.get(&event).map(Arc::clone)
    }

    /// Forgets every hook, as a dispatcher restored from a snapshot holds none.
    #[cfg(any(test, feature = "test-support"))]
    fn clear(&self) {
        self.hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// How many hooks are kept.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.hooks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Whether the server of `service` made the event `event` is an end of.
fn made_by(scheduler: &Scheduler, service: ServiceId, event: ObjectId) -> bool {
    let ipc = scheduler.ipc();
    ipc.service(service)
        .is_some_and(|server| ipc.is_server_event(server, event))
}

impl Dispatcher {
    /// Attaches `hook` to `event` again after a restore (kernel spec, "Waiting and waking": a
    /// hook "is not snapshotted", and "the owning service re-attaches its hooks in its restore
    /// step, checked against the restored events, before the guest runs again"; owner's ruling
    /// of 2026-11-17). Run on the restoring thread, outside any request.
    ///
    /// The check is against what the kernel holds: `event` must be a live writable end of an
    /// event ([`Object::Event`]) that the server of `service` made, which only
    /// [`IpcContext::create_hooked_event`](super::IpcContext::create_hooked_event) makes. Any
    /// other id (a readable end, an event the guest or another server made, one that names
    /// nothing) and a service with no server are [`KernelError::InvalidHandle`], and nothing is
    /// attached. A hook already on `event` is replaced.
    pub fn reattach_event_hook(
        &self,
        service: ServiceId,
        event: ObjectId,
        hook: Arc<dyn EventHook>,
    ) -> SvcResult<()> {
        self.with::<(&Scheduler,), _>(|(scheduler,)| {
            let writable = matches!(scheduler.objects().get(event), Ok(Object::Event(_)));
            if !made_by(scheduler, service, event) || !writable {
                return Err(KernelError::InvalidHandle);
            }
            self.hooks.attach(scheduler.objects(), event, hook);
            Ok(())
        })
    }

    /// Whether `event` is a live end of an event (either end: [`Object::Event`] or
    /// [`Object::ReadableEvent`]) that the server of `service` made, through
    /// [`IpcContext::create_event`](super::IpcContext::create_event) or
    /// [`IpcContext::create_hooked_event`](super::IpcContext::create_hooked_event): how a
    /// service's restore step checks an event its snapshot names. False for any other id (an
    /// event the guest or another server made, one that names nothing) and for a service with no
    /// server. Run on the restoring thread, outside any request.
    #[must_use]
    pub fn is_service_event(&self, service: ServiceId, event: ObjectId) -> bool {
        self.with::<(&Scheduler,), _>(|(scheduler,)| {
            let live = matches!(
                scheduler.objects().get(event),
                Ok(Object::Event(_) | Object::ReadableEvent(_))
            );
            live && made_by(scheduler, service, event)
        })
    }

    /// The transfer memory `object` names: [`KernelError::InvalidHandle`] when it names no live
    /// transfer memory, or one whose pages are not locked (one being made, which no handle names
    /// yet). What a server's request reaches through
    /// [`IpcContext::transfer_memory`](super::IpcContext::transfer_memory), and what a service's
    /// restore step checks a transfer memory its snapshot names against, outside any request.
    pub fn transfer_memory(&self, object: ObjectId) -> SvcResult<TransferMemory> {
        let found = self.with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().get(object));
        match found {
            Ok(Object::TransferMemory(transfer)) if transfer.is_locked() => Ok(transfer),
            _ => Err(KernelError::InvalidHandle),
        }
    }

    /// Forgets every hook, as a dispatcher restored from a snapshot holds none (hooks are code,
    /// not snapshotted): for a service's tests of its restore step. Only with the
    /// `test-support` feature (or in this crate's tests), which any crate may turn on.
    #[cfg(any(test, feature = "test-support"))]
    pub fn forget_event_hooks(&self) {
        self.hooks.clear();
    }

    /// Runs the hook attached to `event`, if it has one: called by the `SignalEvent` arm once
    /// the call has succeeded and the kernel lock is released.
    pub(super) fn run_event_hook(&self, event: ObjectId) {
        // The map's lock goes before the hook runs.
        let hook = self.hooks.get(event);
        if let Some(hook) = hook {
            hook.signalled(event);
        }
    }
}
