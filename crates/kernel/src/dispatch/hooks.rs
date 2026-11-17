//! The dispatcher's side of [`EventHook`]: which hook each hooked event has, and running a hook
//! after a guest's `SignalEvent` (`crate::ipc::hook` says exactly when).
//!
//! The hooks are kept outside the tables, as servers' handlers are: they are code, not state a
//! snapshot can hold, and they run without the kernel lock. They have a lock of their own,
//! which is only ever taken alone or inside the kernel lock (never the other way round), and
//! is held only to read or change the map, never while a hook runs.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use super::Dispatcher;
use crate::id::ObjectId;
use crate::ipc::EventHook;
use crate::object::Objects;

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
