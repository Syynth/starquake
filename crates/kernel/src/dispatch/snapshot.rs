//! The dispatcher's side of the kernel object snapshot ([`crate::object::snapshot`]): saving the
//! object table with the handle table, address space and hooked events it holds
//! ([`Dispatcher::save_objects`]), checking a snapshot against the dispatcher's tables
//! ([`Dispatcher::restore_objects`]), and saying which hooks a restored snapshot names that no
//! service has re-attached yet ([`Dispatcher::unattached_event_hooks`]): the kernel half of the
//! owner's ruling of 2026-11-17 ("Event hooks are re-attached by their owner after a restore").
//!
//! The hooks are the dispatcher's, outside the tables (`dispatch/hooks.rs`), and only the hooked
//! events are saved, each with the service whose server made it: the server that noted it
//! ([`crate::ipc::Ipc::is_server_event`]), found by walking the servers in id order.

use super::Dispatcher;
use crate::handle::HandleTable;
use crate::id::ObjectId;
use crate::object::{HookedEvent, Object, ObjectSnapshotError, Objects, RestoredObjects};
use crate::tables::{AddressSpace, Scheduler};

/// The hooked events among `events` (the dispatcher's hooked ids, in id order), each with the
/// service whose server made it. An id that names no live writable end of an event is left
/// out: its hook can never run again. A live one that no server noted is
/// [`ObjectSnapshotError::HookWithoutServer`].
fn hooked_events(
    scheduler: &Scheduler,
    events: Vec<ObjectId>,
) -> Result<Vec<HookedEvent>, ObjectSnapshotError> {
    let ipc = scheduler.ipc();
    let mut hooked = Vec::new();
    for event in events {
        if !matches!(scheduler.objects().get(event), Ok(Object::Event(_))) {
            continue;
        }
        let service = ipc
            .servers()
            .find(|server| ipc.is_server_event(*server, event))
            .and_then(|server| ipc.server_service(server).ok())
            .ok_or(ObjectSnapshotError::HookWithoutServer { event })?;
        hooked.push(HookedEvent { event, service });
    }
    Ok(hooked)
}

impl Dispatcher {
    /// The object table, the process's handle table and the hooked events as a kernel object
    /// snapshot ([`crate::object::Objects::save`], with the scheduler as its
    /// [`crate::object::KernelView`] and the process's address space for its lends), under the
    /// lock: every table it reads at once, between calls. A failure is a bug in the kernel.
    pub fn save_objects(&self) -> Result<Vec<u8>, ObjectSnapshotError> {
        let locked = self.lock::<(&HandleTable, &AddressSpace, &Scheduler)>();
        let tables = &locked.state.tables;
        let own = tables.processes.own();
        // The hooks' lock is taken inside the kernel lock, the one order it allows.
        let hooked = hooked_events(&tables.scheduler, self.hooks.hooked())?;
        tables.scheduler.objects().save(
            &[(own, &tables.handles)],
            &hooked,
            &tables.scheduler,
            &[(own, &tables.address_space)],
        )
    }

    /// The tables a kernel object snapshot holds ([`crate::object::Objects::restore`] and
    /// [`crate::object::PendingObjects::finish`]), checked against this dispatcher's tables
    /// under the lock: its scheduler as the [`crate::object::KernelView`], and its process's
    /// address space for the lends. Nothing is installed: the scheduler and IPC tables have no
    /// snapshot yet, so the restored tables can be put in place only beside the tables they
    /// were saved with, and the hooks are left as they are.
    pub fn restore_objects(&self, bytes: &[u8]) -> Result<RestoredObjects, ObjectSnapshotError> {
        let locked = self.lock::<(&AddressSpace, &Scheduler)>();
        let tables = &locked.state.tables;
        let own = tables.processes.own();
        Objects::restore(bytes, &tables.scheduler)?.finish(&[(own, &tables.address_space)])
    }

    /// Of `hooked`, the hooked events a restored snapshot names
    /// ([`crate::object::RestoredObjects::hooked`]), those with no hook attached now, in the
    /// order given: what must be empty before the guest runs again (kernel spec, "Waiting and
    /// waking": the owning service re-attaches its hooks "before the guest runs again").
    #[must_use]
    pub fn unattached_event_hooks(&self, hooked: &[HookedEvent]) -> Vec<HookedEvent> {
        let attached = self.hooks.hooked();
        hooked
            .iter()
            .filter(|hook| attached.binary_search(&hook.event).is_err())
            .copied()
            .collect()
    }
}
