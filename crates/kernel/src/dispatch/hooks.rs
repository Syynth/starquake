//! The dispatcher's side of [`EventHook`]: which hook each hooked event has, and running a hook
//! after a guest's `SignalEvent` (`crate::ipc::hook` says exactly when).
//!
//! The hooks are kept outside the tables, as servers' handlers are: they are code, not state a
//! snapshot can hold, and they run without the kernel lock. They have a lock of their own,
//! which is only ever taken alone or inside the kernel lock (never the other way round), and
//! is held only to read or change the map, never while a hook runs.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use starquake_trace::ServiceId;

use super::Dispatcher;
use crate::id::ObjectId;
use crate::ipc::EventHook;
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

    /// The hook attached to `event`, if any.
    fn get(&self, event: ObjectId) -> Option<Arc<dyn EventHook>> {
        let hooks = self.hooks.lock().unwrap_or_else(PoisonError::into_inner);
        hooks.get(&event).map(Arc::clone)
    }

    /// Forgets every hook, as a dispatcher restored from a snapshot holds none.
    #[cfg(test)]
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
            let ipc = scheduler.ipc();
            let made = ipc
                .service(service)
                .is_some_and(|server| ipc.is_server_event(server, event));
            let writable = matches!(scheduler.objects().get(event), Ok(Object::Event(_)));
            if !made || !writable {
                return Err(KernelError::InvalidHandle);
            }
            self.hooks.attach(scheduler.objects(), event, hook);
            Ok(())
        })
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
