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
//! ([`Dispatcher::transfer_memory`]). The copies in and out of a lent transfer memory
//! ([`Dispatcher::read_transfer_memory`], [`Dispatcher::write_transfer_memory`]) live beside
//! that lookup, so a request and a host turn outside any request check the same bounds.
//!
//! # A server's own reference to a lent transfer memory
//!
//! A request lends its server a transfer memory only for as long as the request holds its
//! handles. A server that goes on using it after the guest has closed its own handle (gsv's
//! shared syncpoint page) keeps a reference of its own, as Eden PR #4588's
//! `IGraphicsSupervisor::BindSyncpointMemory` does with `KTransferMemory::Open`, and drops it
//! with `Close` when it is done (`~IGraphicsSupervisor`;
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97). Here a
//! handler keeps one with
//! [`IpcContext::keep_transfer_memory`](super::IpcContext::keep_transfer_memory), and the
//! kernel notes it against the server ([`crate::ipc::Ipc::add_kept`]); the server drops it
//! during a request
//! ([`IpcContext::release_kept_transfer_memory`](super::IpcContext::release_kept_transfer_memory))
//! or on its own thread ([`Dispatcher::release_kept_transfer_memory`]). Only a reference the
//! server's notes hold is dropped, so a server can never drop the guest's or another server's
//! (`only_a_kept_reference_is_released`). The last reference to go tears the transfer memory
//! down as closing the guest's last handle does: its owner's pages are unlocked before the
//! kernel lock is let go (`Locked`'s `Drop`, in `dispatch.rs`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use starquake_memory::GuestMemory;
use starquake_trace::ServiceId;

use super::Dispatcher;
use crate::handle::ObjectRefs;
use crate::id::{ObjectId, ServerId};
use crate::ipc::EventHook;
use crate::memory::TransferMemory;
use crate::memory::shared::span_within;
use crate::object::{Object, Objects};
use crate::result::{CallResult, KernelError, SvcError, SvcResult};
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

/// The live transfer memory `object` names in `scheduler`'s object table, with its owner's pages
/// locked: [`KernelError::InvalidHandle`] for any other id ([`Dispatcher::transfer_memory`]).
fn locked_transfer_memory(scheduler: &Scheduler, object: ObjectId) -> SvcResult<TransferMemory> {
    match scheduler.objects().get(object) {
        Ok(Object::TransferMemory(transfer)) if transfer.is_locked() => Ok(transfer),
        _ => Err(KernelError::InvalidHandle),
    }
}

/// Opens one reference to the transfer memory `object` names for `server` to keep, and notes it
/// against the server (module docs, "A server's own reference"). [`KernelError::InvalidHandle`],
/// with nothing opened, unless `object` names a live transfer memory whose owner's pages are
/// locked. The note is made after the reference is opened; if it cannot be made (a server the
/// table does not know: the kernel's own bookkeeping), the reference is dropped again and the
/// error returned, so no reference is left that nobody holds.
pub(super) fn keep_transfer_memory_in(
    scheduler: &mut Scheduler,
    server: ServerId,
    object: ObjectId,
) -> CallResult<TransferMemory> {
    let transfer = locked_transfer_memory(scheduler, object)?;
    scheduler
        .objects_mut()
        .open(object)
        .map_err(|error| SvcError::internal(&error))?;
    if let Err(error) = scheduler.ipc_mut().1.add_kept(server, object) {
        scheduler.close_object(object)?;
        return Err(error.into());
    }
    Ok(transfer)
}

/// Drops one reference `server` keeps to `object` (module docs, "A server's own reference"):
/// [`KernelError::InvalidHandle`], with nothing changed, when the server's notes hold none. When
/// it was the object's last reference the transfer memory goes, and its owner's pages are
/// unlocked when the kernel lock is let go.
pub(super) fn release_kept_transfer_memory_in(
    scheduler: &mut Scheduler,
    server: ServerId,
    object: ObjectId,
) -> CallResult<()> {
    if !scheduler.ipc_mut().1.take_kept(server, object) {
        return Err(KernelError::InvalidHandle.into());
    }
    scheduler.close_object(object)
}

impl Dispatcher {
    /// Drops one reference the server of `service` keeps to the transfer memory `object` names
    /// (module docs, "A server's own reference"), on the server's own thread, outside any
    /// request: how gsv lets the shared syncpoint page's memories go once the root's last
    /// session has closed. [`KernelError::InvalidHandle`], with nothing changed, when that
    /// server keeps no reference to `object` (it never kept one, it has dropped every one it
    /// kept, or no server is known as `service`). When it was the last reference the transfer
    /// memory goes, as closing the guest's last handle makes it go: its owner's pages are
    /// unlocked before the kernel lock is let go. A failure of the kernel's own bookkeeping is
    /// logged, and [`KernelError::InvalidState`] here.
    pub fn release_kept_transfer_memory(
        &self,
        service: ServiceId,
        object: ObjectId,
    ) -> SvcResult<()> {
        self.with::<(&mut Scheduler,), _>(|(scheduler,)| {
            let server = scheduler
                .ipc()
                .service(service)
                .ok_or(KernelError::InvalidHandle)?;
            release_kept_transfer_memory_in(scheduler, server, object)
        })
        .map_err(|error| match error {
            SvcError::Kernel(error) => error,
            SvcError::Internal(what) => {
                tracing::error!(
                    what,
                    "kernel bookkeeping failed releasing a kept transfer memory"
                );
                KernelError::InvalidState
            }
        })
    }

    /// How many references the server of `service` keeps to `object` (module docs, "A server's
    /// own reference"): how a service's restore step checks a transfer memory its snapshot says
    /// it keeps, and how tests count them. Zero for an object it keeps none of and for a service
    /// with no server.
    #[must_use]
    pub fn kept_transfer_memory(&self, service: ServiceId, object: ObjectId) -> u32 {
        self.with::<(&Scheduler,), _>(|(scheduler,)| {
            let ipc = scheduler.ipc();
            ipc.service(service)
                .map_or(0, |server| ipc.kept(server, object))
        })
    }

    /// How many references the server of `service` keeps in all, over every transfer memory
    /// (module docs, "A server's own reference"; [`crate::ipc::Ipc::kept_total`]): how a
    /// service's restore step checks that its snapshot accounts for every reference the
    /// server keeps. Zero for a service with no server.
    #[must_use]
    pub fn kept_transfer_memories(&self, service: ServiceId) -> u32 {
        self.with::<(&Scheduler,), _>(|(scheduler,)| {
            let ipc = scheduler.ipc();
            ipc.service(service)
                .map_or(0, |server| ipc.kept_total(server))
        })
    }

    /// How many sessions on the server of `service` have their server end open
    /// ([`crate::ipc::Ipc::open_sessions`], which leaves out a session whose request in flight
    /// is a close message, as the server has already forgotten it): zero for a service with no
    /// server. Run outside any
    /// request; gsv asks it on its timer thread, before it lets the shared syncpoint page go,
    /// whether a root session has opened since the last one closed.
    #[must_use]
    pub fn open_sessions(&self, service: ServiceId) -> usize {
        self.with::<(&Scheduler,), _>(|(scheduler,)| {
            let ipc = scheduler.ipc();
            ipc.service(service)
                .map_or(0, |server| ipc.open_sessions(server))
        })
    }

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
        self.with::<(&Scheduler,), _>(|(scheduler,)| locked_transfer_memory(scheduler, object))
    }

    /// Copies `out.len()` bytes at `offset` into the transfer memory `object` names into `out`,
    /// through `memory`, the owner's guest memory: the one check a server's request
    /// ([`IpcContext::read_transfer_memory`](super::IpcContext::read_transfer_memory)) and a
    /// host turn outside any request make. [`KernelError::InvalidHandle`] as for
    /// [`Dispatcher::transfer_memory`], and [`KernelError::InvalidCurrentMemory`] unless the
    /// bytes lie wholly inside the transfer memory (`span_within`) and `memory` reads them;
    /// nothing is copied then.
    pub fn read_transfer_memory(
        &self,
        memory: &dyn GuestMemory,
        object: ObjectId,
        offset: u64,
        out: &mut [u8],
    ) -> SvcResult<()> {
        let address = self.lent_range(object, offset, out.len())?;
        memory
            .read(address, out)
            .map_err(|_| KernelError::InvalidCurrentMemory)
    }

    /// Copies `data` into the transfer memory `object` names, at `offset`, through `memory`.
    /// The same rules as [`Dispatcher::read_transfer_memory`].
    pub fn write_transfer_memory(
        &self,
        memory: &dyn GuestMemory,
        object: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        let address = self.lent_range(object, offset, data.len())?;
        memory
            .write(address, data)
            .map_err(|_| KernelError::InvalidCurrentMemory)
    }

    /// The owner's address of `len` bytes at `offset` into the transfer memory `object` names,
    /// if they lie wholly inside it.
    fn lent_range(&self, object: ObjectId, offset: u64, len: usize) -> SvcResult<u64> {
        let transfer = self.transfer_memory(object)?;
        span_within(transfer.address(), transfer.size(), offset, len)
            .ok_or(KernelError::InvalidCurrentMemory)
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
