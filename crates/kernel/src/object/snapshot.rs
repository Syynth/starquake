//! The object table and each process's handle table saved and restored ([`Objects::save`],
//! [`Objects::restore`]): the observability spec's snapshots, step 6 ("kernel and memory
//! first"), with the kernel's shared header ([`crate::snapshot`]).
//!
//! # What is saved
//!
//! Every row of the object table ([`Object`]: threads, processes, sessions, both ends of
//! events, shared and transfer memory, resource limits) with its reference count, in its slot under its
//! generation, with the table's free slots in the order they are reused and its slot limit, so
//! every id the other tables hold names the same object after a restore and the next object
//! made gets the same id. Each process's handle table, whole ([`HandleTable`]: its entries, free
//! list, next linear id and high-water mark). And which events are hooked, by which service
//! ([`HookedEvent`]): the kernel half of the owner's ruling of 2026-11-17 ("Event hooks are
//! re-attached by their owner after a restore"; kernel spec, "Waiting and waking"). A hook is
//! code, so nothing in the bytes names one: the owning service re-attaches its hook
//! ([`crate::dispatch::Dispatcher::reattach_event_hook`]) in its restore step, and
//! [`crate::dispatch::Dispatcher::unattached_event_hooks`] says which saved hooks are still
//! missing before the guest runs again.
//!
//! Not saved: the transfer memories released while their pages were locked
//! ([`Objects::take_unlocks`]). The dispatcher unlocks them before it lets the kernel lock go,
//! and a snapshot is taken between calls, so there are none; [`Objects::save`] refuses a table
//! that has some ([`ObjectSnapshotError::PendingUnlocks`]). Nor the shared memories released
//! whose pages are still to be given back ([`Objects::take_frees`]), for the same reason:
//! [`Objects::save`] refuses a table that has some ([`ObjectSnapshotError::PendingFrees`]),
//! checked after the unlocks. Not saved because it is not the
//! object table's: what a thread, process or session's own row holds (the scheduler and IPC
//! tables), a shared memory's pages (guest memory) and the address spaces.
//!
//! # Format
//!
//! [`crate::snapshot`]'s header, with [`OBJECTS_MAGIC`] and [`OBJECTS_VERSION`], then the body,
//! in this order: the object table as a [`Slab`] (its slots, each vacant with its generation,
//! occupied with its generation and row, or retired; the free stack; the slot limit; the live
//! count), each row its object then its reference count; the handle tables (each its process,
//! then the table's entries, free list, next linear id and high-water mark); the hooked events
//! (each the writable end, then the service). An object is a tag (`SavedObject`'s variant
//! index) then its fields: a thread or process its id; a session nothing; an event's writable
//! end its readable end; a readable end whether it is signalled; a shared memory the base and
//! size of its pages and its user permission's bits; a transfer memory its owner, address,
//! size, the permission bits its owner keeps, and whether its pages are locked; a resource limit
//! its process (version 2 added it; version 1 had no resource limits). Permissions are
//! the bits of [`SvcPermission::bits`], the reference's `Svc::MemoryPermission`. The layout is
//! starquake's own: Horizon has no snapshot to follow. The golden-bytes test
//! `the_body_layout_is_pinned` pins it with every object kind; changing it means bumping
//! [`OBJECTS_VERSION`].
//!
//! # Checks
//!
//! A snapshot is a file and is not trusted. The relations to the tables beside the object
//! table are asked of a [`KernelView`] (the production one is the [`Scheduler`], which holds
//! the thread, process and IPC tables). [`Objects::restore`] checks, in this order, and returns
//! the first failure ([`ObjectSnapshotError`]):
//!
//! 1. The header and the body's encoding ([`crate::snapshot::decode`]: header, magic, version,
//!    body, trailing bytes). The object table's slot layout is checked as it decodes, by
//!    [`Slab`]'s own deserialisation (free stack in range, vacant and without repeats,
//!    generations not zero, counts agreeing), whose failure is a body that does not decode
//!    ([`crate::snapshot::FormatError::Body`]; the slab's tests pin each case).
//! 2. Each handle table in turn: its process comes after the one before
//!    ([`ObjectSnapshotError::HandleOrder`]), is a process the kernel has
//!    ([`ObjectSnapshotError::HandleProcess`]), and its layout is one a handle table can have
//!    ([`ObjectSnapshotError::HandleLayout`], [`HandleLayoutError`]'s checks).
//! 3. Each object in slot order ([`ObjectSnapshotError::Object`], first failing check of the
//!    first failing object): it has a reference ([`ObjectFault::NoReferences`]), then by kind:
//!    a thread's row names this object back ([`ObjectFault::Thread`]); a process's row names
//!    this object back ([`ObjectFault::Process`]); a session is one the IPC table holds
//!    ([`ObjectFault::Session`]); an event's readable end is a live readable end or names
//!    nothing live ([`ObjectFault::Readable`]), is an id the object table has issued, live or
//!    freed since, not a slot past its end or a generation its slot has not reached
//!    ([`ObjectFault::ReadableNotIssued`]), and no earlier event's ([`ObjectFault::SharedReadable`]);
//!    a shared memory has pages ([`starquake_memory::BackingRange`]: whole pages, not empty, not
//!    wrapping) and a shared memory permission ([`ObjectFault::SharedMemory`]), and its pages
//!    share none with an earlier one's ([`ObjectFault::SharedOverlap`]); a transfer memory's
//!    owner keeps a permission a lend leaves ([`ObjectFault::TransferPermission`]), its pages are
//!    locked ([`ObjectFault::Unlocked`]), and its owner has a handle table in the snapshot
//!    ([`ObjectFault::Owner`]); a resource limit's process's row names this object as its
//!    limit ([`ObjectFault::ResourceLimit`]), so it belongs to a process the kernel has and no
//!    two limits share a process.
//! 4. Each handle of each table (processes in order, lowest index first): it names a live
//!    object ([`HandleFault::Stale`]), and its object has a reference for it and for each
//!    handle to it before it ([`HandleFault::Overused`]): each handle holds one reference
//!    (`KHandleTable::Add` opens one), so an object with `n` references can have at most `n`
//!    handles across the tables. A resource limit's process row also holds one (check 3 found
//!    it names the limit), so a limit with `n` references can have at most `n - 1` handles: the
//!    `n`th is [`HandleFault::LimitRow`], reported after [`HandleFault::Overused`] for the same
//!    handle.
//! 5. Each hooked event in turn ([`ObjectSnapshotError::Hook`]): it comes after the one before
//!    ([`HookFault::Order`]), is a live writable end of an event ([`HookFault::NotAnEvent`]),
//!    names a service the kernel has a server for ([`HookFault::UnknownService`]), and that
//!    server made it ([`HookFault::NotMadeBy`]).
//!
//! What [`Objects::restore`] returns is [`PendingObjects`]: the tables are handed over only by
//! [`PendingObjects::finish`], once the address spaces are restored (which need the shared
//! memories, [`PendingObjects::shared_memory`]), after the last check:
//!
//! 6. The address spaces given are in process order without repeats
//!    ([`ObjectSnapshotError::SpaceOrder`]), and there is one exactly for each process with a
//!    handle table ([`ObjectSnapshotError::MissingSpace`], [`ObjectSnapshotError::ExtraSpace`],
//!    the first process that differs). Then each, in process order, has its locked memory
//!    checked against the transfer memories its process owns ([`AddressSpace::check_lends`],
//!    [`ObjectSnapshotError::Lends`]).
//!
//! [`Objects::save`] runs checks 2 to 6 on what it is about to write: live tables that fail one
//! are a bug in the kernel, which it reports rather than writes.
//!
//! # Why each relation holds live
//!
//! - **References.** A row with no reference does not exist: [`Objects::release`] removes the
//!   row with its last one.
//! - **Threads and processes.** [`Scheduler::create_user_thread`] records the object it makes
//!   in the thread's row, and the thread's row goes only when the object's last reference does
//!   ([`Scheduler::tear_down`]). [`Scheduler::add_process`] records the process's object in
//!   its row, and process rows are never removed.
//! - **Resource limits.** [`Scheduler::add_process`] makes the process's limit beside its
//!   object and records it in the row, which holds a reference to it and is never removed. No
//!   other kernel path makes one, but the compiler does not enforce it:
//!   [`ResourceLimit::of`] and [`Objects::create`] are public, so code in another crate can
//!   make a second limit for a process. Its process row names the first, so [`Objects::save`]
//!   refuses that state with [`ObjectFault::ResourceLimit`], as restore would.
//! - **Sessions.** [`crate::ipc::Ipc::open_session`] makes the object and the session's row
//!   together, and the row goes when the object's last reference does
//!   ([`crate::ipc::Ipc::forget`], [`crate::ipc::Ipc::discard_session`]).
//! - **Events.** [`Scheduler::create_event`] makes each writable end with a readable end of its
//!   own. The writable end holds no reference to it, so the readable end may be gone; a slot is
//!   reused only under a new generation, so the id then names nothing live, and it is an id the
//!   table issued, so no later object gets it.
//! - **Shared memory.** [`SharedMemory::new`] refuses another permission; its pages come from
//!   the address space's [`crate::memory::Mapper::allocate_shared`], which gives each shared
//!   memory pages of its own: a live shared memory's pages are its own. They are given back
//!   ([`crate::memory::Mapper::release_shared`]) only after its last reference went, and the
//!   translation hands them out again only once every flush their unmappings owe has run
//!   (`memory::shared`, "The pages' lifetime").
//! - **Transfer memory.** `CreateTransferMemory` checks the permission
//!   ([`crate::memory::is_transfer_memory_permission`]); an object whose pages are not locked
//!   lives only inside that call, which closes it on every path that fails before the lock
//!   ([`crate::svc::transfer_memory`]); its owner is the calling process, whose handle table
//!   gets the handle.
//! - **Handles.** [`HandleTable::add`] opens one reference and [`HandleTable::remove`] closes
//!   it, so each handle holds one.
//! - **Hooks.** Only `create_hooked_event` and `reattach_event_hook` attach a hook, each to a
//!   writable end its service's server made (`crate::dispatch`), and the server keeps its
//!   creator's reference, so the event stays live. A hook whose event is gone is left out of
//!   the saved list.
//!
//! The round trips `a_live_kernel_saves_restores_and_saves_the_same_bytes` and
//! `a_space_held_by_a_lend_and_handles_of_every_kind_round_trip_with_a_second_process` (in
//! `object/snapshot/tests.rs`) save live tables with every object kind, and
//! `tables_built_by_random_operations_save_and_restore_as_they_were` saves and restores after
//! each of random events, lends, signals, second handles and closes.
//!
//! # Not checked here
//!
//! - **The other direction of each relation**: that every thread and process row's object, every
//!   process row's resource limit, and
//!   every session the IPC table holds, is a row of the restored table of the right kind, and
//!   that every event a server noted is live. Those are the scheduler's and the IPC table's
//!   own restores, not built yet; until then [`KernelView`] is the live scheduler.
//! - **A resource limit's values.** The object holds none ([`crate::resource_limit`], "No
//!   counts of its own"): its physical memory values are the address space's count, which the
//!   address space's restore checks (reserved within the pool and equal to what the blocks
//!   use), and its thread count is the owner's thread rows, the scheduler's restore, not built
//!   yet. A forged snapshot cannot give a limit other limit values, as none are saved.
//! - **References beyond the handles'.** Check 4 counts the handles and, for a resource limit,
//!   its process row's reference: a thread's reference to itself, a process row's reference to
//!   its process object, a server's creator reference to the events it made, a request's to its
//!   session and a wait's are other tables' and are not counted, so a count higher than the
//!   holders explain is not refused.
//! - **A shared memory's pages against the mapper**: that its pages are pages the mapper
//!   allocated. The mapper is rebuilt by the address space's restore and keeps no record of
//!   them.
//! - **A transfer memory's range**: its shape and where it lies are check 6's
//!   ([`crate::memory::LendFault::Shape`]).
//! - **That an event's live readable end was made with it.** Check 3 refuses a readable end
//!   that an earlier event also names, comparing only the events in the snapshot. An event that
//!   names a live readable end made with another event whose writable end is gone is accepted:
//!   live code cannot leave this ([`Scheduler::create_event`] makes each writable end with its
//!   own readable end), but a forged snapshot can, and the guest's signals through that event
//!   then wake the waiters of the other event's readable end.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};
use starquake_memory::BackingRange;
use starquake_trace::ServiceId;

use super::{Object, Objects, Row};
use crate::handle::{Handle, HandleLayoutError, HandleTable, HandleTableData};
use crate::id::{ObjectId, ProcessId, SlabId, ThreadId};
use crate::memory::{
    AddressSpace, LendError, SharedMemory, SvcPermission, TransferMemory,
    is_transfer_memory_permission,
};
use crate::process::ProcessRecord;
use crate::resource_limit::ResourceLimit;
use crate::sched::{Scheduler, Thread};
use crate::slab::Slab;
use crate::snapshot::{Format, FormatError, decode, encode};
use crate::sync::{Event, ReadableEvent};

#[cfg(test)]
mod tests;

/// First bytes of every kernel object snapshot.
pub const OBJECTS_MAGIC: [u8; 4] = *b"SQOB";

/// Layout version written into every kernel object snapshot and required when reading one.
pub const OBJECTS_VERSION: u32 = 2;

/// The kernel object snapshot's header.
const FORMAT: Format = Format {
    name: "kernel object",
    magic: OBJECTS_MAGIC,
    version: OBJECTS_VERSION,
};

/// An event a service hooked: its writable end, and the service whose server made it and
/// re-attaches the hook after a restore. Plain data: the restore checks it against the kernel
/// (module docs, check 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HookedEvent {
    /// The event's writable end, what the guest signals.
    pub event: ObjectId,
    /// The service whose server made it.
    pub service: ServiceId,
}

/// What the object table's restore asks of the tables beside it: the scheduler's thread and
/// process rows, and the IPC table's sessions and servers. [`Scheduler`] is the production
/// one.
pub trait KernelView {
    /// The object the thread's row names, if the thread exists and has one.
    fn thread_object(&self, thread: ThreadId) -> Option<ObjectId>;
    /// The object the process's row names, if the process exists.
    fn process_object(&self, process: ProcessId) -> Option<ObjectId>;
    /// The resource limit object the process's row names, if the process exists.
    fn resource_limit_object(&self, process: ProcessId) -> Option<ObjectId>;
    /// Whether the IPC table holds a session whose client end is `object`.
    fn is_session(&self, object: ObjectId) -> bool;
    /// Whether a server is known as `service`.
    fn has_service(&self, service: ServiceId) -> bool;
    /// Whether the server known as `service` made the event `event` is an end of.
    fn made_by(&self, service: ServiceId, event: ObjectId) -> bool;
}

impl KernelView for Scheduler {
    fn thread_object(&self, thread: ThreadId) -> Option<ObjectId> {
        self.thread(thread).ok().and_then(Thread::object)
    }

    fn process_object(&self, process: ProcessId) -> Option<ObjectId> {
        self.process(process).ok().map(ProcessRecord::object)
    }

    fn resource_limit_object(&self, process: ProcessId) -> Option<ObjectId> {
        self.process(process)
            .ok()
            .map(ProcessRecord::resource_limit)
    }

    fn is_session(&self, object: ObjectId) -> bool {
        self.ipc().is_session(object)
    }

    fn has_service(&self, service: ServiceId) -> bool {
        self.ipc().service(service).is_some()
    }

    fn made_by(&self, service: ServiceId, event: ObjectId) -> bool {
        let ipc = self.ipc();
        ipc.service(service)
            .is_some_and(|server| ipc.is_server_event(server, event))
    }
}

/// An id in plain words: its kind, slot and generation.
fn words(kind: &str, index: u32, generation: u32) -> String {
    format!("{kind} {index} (generation {generation})")
}

/// What went wrong with one object ([`ObjectSnapshotError::Object`]), in the order the checks
/// are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ObjectFault {
    /// Its reference count is zero.
    #[error("has no references")]
    NoReferences,
    /// It is a thread whose row does not exist or names another object.
    #[error("is a thread whose row does not name it back")]
    Thread,
    /// It is a process whose row does not exist or names another object.
    #[error("is a process whose row does not name it back")]
    Process,
    /// It is a session the IPC table does not hold.
    #[error("is a session the IPC table does not hold")]
    Session,
    /// It is an event whose readable end names a live object that is no readable end.
    #[error("is an event whose readable end is another kind of object")]
    Readable,
    /// It is an event whose readable end names an id the object table has not issued: a slot
    /// past the table's end, or a slot under a generation it has not reached. The next object
    /// made there would get that id, and the event would signal it.
    #[error("is an event whose readable end names an id the object table has not made yet")]
    ReadableNotIssued,
    /// It is an event whose readable end an earlier event names too.
    #[error("is an event whose readable end an earlier event signals too")]
    SharedReadable,
    /// It is a shared memory whose pages are not whole pages or wrap, or whose permission is
    /// not read or read-write.
    #[error("is a shared memory with pages or a permission no shared memory has")]
    SharedMemory,
    /// It is a shared memory whose pages overlap an earlier one's.
    #[error("is a shared memory whose pages overlap an earlier one's")]
    SharedOverlap,
    /// It is a transfer memory whose owner keeps a permission no lend leaves.
    #[error("is a transfer memory whose owner keeps a permission no lend leaves")]
    TransferPermission,
    /// It is a transfer memory whose pages are not locked.
    #[error("is a transfer memory whose pages are not locked")]
    Unlocked,
    /// It is a transfer memory of a process the snapshot holds no handle table for.
    #[error("is a transfer memory of a process the snapshot holds no handle table for")]
    Owner,
    /// It is a resource limit whose process's row does not exist or names another limit.
    #[error("is a resource limit whose process's row does not name it back")]
    ResourceLimit,
}

/// What went wrong with one handle ([`ObjectSnapshotError::Handle`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HandleFault {
    /// It names no live object.
    #[error("names no live object")]
    Stale,
    /// Its object has fewer references than the handles to it so far.
    #[error("is one handle more to its object than the object has references")]
    Overused,
    /// Its object is a resource limit, and this handle is one more than its references leave
    /// once the process row that names the limit (check 3) holds its own.
    #[error("takes the reference its resource limit's process row holds")]
    LimitRow,
}

/// What went wrong with one hooked event ([`ObjectSnapshotError::Hook`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HookFault {
    /// It does not come after the one before.
    #[error("is out of order or repeats the one before it")]
    Order,
    /// It names no live writable end of an event.
    #[error("names no live writable end of an event")]
    NotAnEvent,
    /// No server is known as its service.
    #[error("names a service that has no server")]
    UnknownService,
    /// Its service's server did not make it.
    #[error("names an event its service's server did not make")]
    NotMadeBy,
}

/// Why the object table and handle tables could not be saved or restored. See the module docs,
/// "Checks", for the order.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObjectSnapshotError {
    /// The header or the body's encoding ([`crate::snapshot::decode`]).
    #[error(transparent)]
    Format(#[from] FormatError),
    /// The table holds released transfer memories whose pages are still to be unlocked: a
    /// snapshot is taken between calls, when it holds none.
    #[error(
        "released transfer memories whose pages are still locked: {count} (a snapshot is taken between calls, when there are none)"
    )]
    PendingUnlocks {
        /// How many.
        count: usize,
    },
    /// The table holds released shared memories whose pages are still to be given back: a
    /// snapshot is taken between calls, when it holds none.
    #[error(
        "released shared memories whose pages are still to be given back: {count} (a snapshot is taken between calls, when there are none)"
    )]
    PendingFrees {
        /// How many.
        count: usize,
    },
    /// A live event has a hook but no server noted it as its own, so no service can re-attach
    /// the hook after a restore. Only a save reports it, from the dispatcher's hooks
    /// ([`crate::dispatch::Dispatcher::save_objects`]).
    #[error(
        "{} has a hook but no server made it",
        words("object", .event.index(), .event.generation())
    )]
    HookWithoutServer {
        /// The event's writable end.
        event: ObjectId,
    },
    /// A handle table's process does not come after the one before.
    #[error(
        "the handle table of {} is out of order or repeats the one before it",
        words("process", .process.index(), .process.generation())
    )]
    HandleOrder {
        /// Its process.
        process: ProcessId,
    },
    /// A handle table's process is not one the kernel has.
    #[error(
        "the handle table of {} belongs to no process the kernel has",
        words("process", .process.index(), .process.generation())
    )]
    HandleProcess {
        /// Its process.
        process: ProcessId,
    },
    /// A handle table's layout is one no handle table has.
    #[error(
        "the handle table of {} is damaged: {error}",
        words("process", .process.index(), .process.generation())
    )]
    HandleLayout {
        /// Its process.
        process: ProcessId,
        /// What is wrong with it.
        error: HandleLayoutError,
    },
    /// An object fails a check.
    #[error(
        "{} {fault}",
        words("object", .object.index(), .object.generation())
    )]
    Object {
        /// The object.
        object: ObjectId,
        /// The first check it fails.
        fault: ObjectFault,
    },
    /// A handle fails a check.
    #[error(
        "handle {:#010x} of {} {fault}",
        .handle.raw(),
        words("process", .process.index(), .process.generation())
    )]
    Handle {
        /// The process whose table holds it.
        process: ProcessId,
        /// The handle.
        handle: Handle,
        /// The first check it fails.
        fault: HandleFault,
    },
    /// A hooked event fails a check.
    #[error(
        "the hook of service {} on {} {fault}",
        .hooked.service.0,
        words("object", .hooked.event.index(), .hooked.event.generation())
    )]
    Hook {
        /// The hooked event.
        hooked: HookedEvent,
        /// The first check it fails.
        fault: HookFault,
    },
    /// An address space was given out of process order, or twice.
    #[error(
        "the address space of {} is out of order or repeats the one before it",
        words("process", .process.index(), .process.generation())
    )]
    SpaceOrder {
        /// Its process.
        process: ProcessId,
    },
    /// No address space was given for a process whose handle table the snapshot holds.
    #[error(
        "no address space was given for {}, whose handle table the snapshot holds",
        words("process", .process.index(), .process.generation())
    )]
    MissingSpace {
        /// The process.
        process: ProcessId,
    },
    /// An address space was given for a process whose handle table the snapshot does not hold.
    #[error(
        "an address space was given for {}, whose handle table the snapshot does not hold",
        words("process", .process.index(), .process.generation())
    )]
    ExtraSpace {
        /// The process.
        process: ProcessId,
    },
    /// A process's locked memory disagrees with the transfer memories it owns.
    #[error(
        "the memory of {} disagrees with its transfer memories: {error}",
        words("process", .process.index(), .process.generation())
    )]
    Lends {
        /// The process.
        process: ProcessId,
        /// What [`AddressSpace::check_lends`] found.
        error: LendError,
    },
}

/// An object as a snapshot holds it: every field plain, every relation checked on restore
/// (module docs, "Checks"). The variant order is the saved tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SavedObject {
    /// [`Object::Thread`].
    Thread(ThreadId),
    /// [`Object::Process`].
    Process(ProcessId),
    /// [`Object::Session`].
    Session,
    /// [`Object::Event`]: its readable end.
    Event { readable: ObjectId },
    /// [`Object::ReadableEvent`].
    ReadableEvent { signalled: bool },
    /// [`Object::SharedMemory`]: its pages and [`SvcPermission::bits`] of its user permission.
    SharedMemory {
        base: u64,
        size: u64,
        user_permission: u8,
    },
    /// [`Object::TransferMemory`], with [`SvcPermission::bits`] of what its owner keeps.
    TransferMemory {
        owner: ProcessId,
        address: u64,
        size: u64,
        owner_permission: u8,
        locked: bool,
    },
    /// [`Object::ResourceLimit`]: the process it belongs to.
    ResourceLimit { owner: ProcessId },
}

impl SavedObject {
    /// The saved form of `object`.
    fn of(object: Object) -> Self {
        match object {
            Object::Thread(thread) => Self::Thread(thread),
            Object::Process(process) => Self::Process(process),
            Object::Session => Self::Session,
            Object::Event(event) => Self::Event {
                readable: event.readable(),
            },
            Object::ReadableEvent(state) => Self::ReadableEvent {
                signalled: state.is_signalled(),
            },
            Object::SharedMemory(memory) => Self::SharedMemory {
                base: memory.backing().base(),
                size: memory.size(),
                user_permission: memory.user_permission().bits(),
            },
            Object::TransferMemory(transfer) => Self::TransferMemory {
                owner: transfer.owner(),
                address: transfer.address(),
                size: transfer.size(),
                owner_permission: transfer.owner_permission().bits(),
                locked: transfer.is_locked(),
            },
            Object::ResourceLimit(limit) => Self::ResourceLimit {
                owner: limit.owner(),
            },
        }
    }
}

/// One row as a snapshot holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct SavedRow {
    object: SavedObject,
    /// Checked to be non-zero on restore ([`ObjectFault::NoReferences`]).
    refs: u32,
}

/// One process's handle table as a snapshot holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SavedHandles {
    process: ProcessId,
    table: HandleTableData,
}

/// The snapshot's body, in its saved order (module docs, "Format").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Body {
    objects: Slab<ObjectId, SavedRow>,
    handles: Vec<SavedHandles>,
    hooked: Vec<HookedEvent>,
}

impl Body {
    /// What a snapshot of these tables holds.
    fn of(
        objects: &Objects,
        handles: &[(ProcessId, &HandleTable)],
        hooked: &[HookedEvent],
    ) -> Self {
        let rows = objects.rows.try_map(|_, row| {
            Ok::<_, core::convert::Infallible>(SavedRow {
                object: SavedObject::of(row.object),
                refs: row.refs.get(),
            })
        });
        let objects = match rows {
            Ok(rows) => rows,
            Err(never) => match never {},
        };
        Self {
            objects,
            handles: handles
                .iter()
                .map(|(process, table)| SavedHandles {
                    process: *process,
                    table: table.data(),
                })
                .collect(),
            hooked: hooked.to_vec(),
        }
    }

    /// Checks 2 to 5 of the module docs, in their order, and the tables they leave.
    fn check(&self, kernel: &dyn KernelView) -> Result<PendingObjects, ObjectSnapshotError> {
        let handles = self.check_handle_tables(kernel)?;
        let owners: BTreeSet<ProcessId> = handles.iter().map(|(process, _)| *process).collect();
        let objects = self.check_objects(kernel, &owners)?;
        check_handles(&objects, &handles)?;
        check_hooked(&objects, &self.hooked, kernel)?;
        Ok(PendingObjects {
            objects,
            handles,
            hooked: self.hooked.clone(),
        })
    }

    /// Check 2: the handle tables, each with its process.
    fn check_handle_tables(
        &self,
        kernel: &dyn KernelView,
    ) -> Result<Vec<(ProcessId, HandleTable)>, ObjectSnapshotError> {
        let mut tables = Vec::new();
        let mut previous: Option<ProcessId> = None;
        for saved in &self.handles {
            let process = saved.process;
            if previous.is_some_and(|before| process <= before) {
                return Err(ObjectSnapshotError::HandleOrder { process });
            }
            if kernel.process_object(process).is_none() {
                return Err(ObjectSnapshotError::HandleProcess { process });
            }
            let table = HandleTable::try_from(saved.table.clone())
                .map_err(|error| ObjectSnapshotError::HandleLayout { process, error })?;
            tables.push((process, table));
            previous = Some(process);
        }
        Ok(tables)
    }

    /// Check 3: each object in slot order, rebuilt as the object table holds it. `owners` are
    /// the processes with a handle table in the snapshot.
    fn check_objects(
        &self,
        kernel: &dyn KernelView,
        owners: &BTreeSet<ProcessId>,
    ) -> Result<Objects, ObjectSnapshotError> {
        let mut readables = BTreeSet::new();
        // Each shared memory's pages so far, as start to end.
        let mut pages: BTreeMap<u64, u64> = BTreeMap::new();
        let rows = self.objects.try_map(|id, saved| {
            let fault = |fault| ObjectSnapshotError::Object { object: id, fault };
            let refs = NonZeroU32::new(saved.refs).ok_or(fault(ObjectFault::NoReferences))?;
            let object = match saved.object {
                SavedObject::Thread(thread) => {
                    if kernel.thread_object(thread) != Some(id) {
                        return Err(fault(ObjectFault::Thread));
                    }
                    Object::Thread(thread)
                }
                SavedObject::Process(process) => {
                    if kernel.process_object(process) != Some(id) {
                        return Err(fault(ObjectFault::Process));
                    }
                    Object::Process(process)
                }
                SavedObject::Session => {
                    if !kernel.is_session(id) {
                        return Err(fault(ObjectFault::Session));
                    }
                    Object::Session
                }
                SavedObject::Event { readable } => {
                    let names = self.objects.get(readable).map(|row| row.object);
                    if names.is_ok_and(|named| !matches!(named, SavedObject::ReadableEvent { .. }))
                    {
                        return Err(fault(ObjectFault::Readable));
                    }
                    if !self.objects.issued(readable) {
                        return Err(fault(ObjectFault::ReadableNotIssued));
                    }
                    if !readables.insert(readable) {
                        return Err(fault(ObjectFault::SharedReadable));
                    }
                    Object::Event(Event::signalling(readable))
                }
                SavedObject::ReadableEvent { signalled } => {
                    Object::ReadableEvent(ReadableEvent::with_signalled(signalled))
                }
                SavedObject::SharedMemory {
                    base,
                    size,
                    user_permission,
                } => {
                    let memory = BackingRange::new(base, size).ok().and_then(|backing| {
                        let permission = SvcPermission::from_raw(u32::from(user_permission))?;
                        SharedMemory::new(backing, permission).ok()
                    });
                    let memory = memory.ok_or(fault(ObjectFault::SharedMemory))?;
                    let (start, end) = (memory.backing().base(), memory.backing().end());
                    let overlaps = pages
                        .range(..end)
                        .next_back()
                        .is_some_and(|(_, before_end)| start < *before_end);
                    if overlaps {
                        return Err(fault(ObjectFault::SharedOverlap));
                    }
                    pages.insert(start, end);
                    Object::SharedMemory(memory)
                }
                SavedObject::TransferMemory {
                    owner,
                    address,
                    size,
                    owner_permission,
                    locked,
                } => {
                    let permission = SvcPermission::from_raw(u32::from(owner_permission))
                        .filter(|permission| is_transfer_memory_permission(*permission))
                        .ok_or(fault(ObjectFault::TransferPermission))?;
                    if !locked {
                        return Err(fault(ObjectFault::Unlocked));
                    }
                    if !owners.contains(&owner) {
                        return Err(fault(ObjectFault::Owner));
                    }
                    let lent = TransferMemory::new(owner, address, size, permission);
                    Object::TransferMemory(lent.with_pages_locked())
                }
                SavedObject::ResourceLimit { owner } => {
                    if kernel.resource_limit_object(owner) != Some(id) {
                        return Err(fault(ObjectFault::ResourceLimit));
                    }
                    Object::ResourceLimit(ResourceLimit::of(owner))
                }
            };
            Ok(Row { object, refs })
        })?;
        Ok(Objects {
            rows,
            unlocks: Vec::new(),
            frees: Vec::new(),
        })
    }
}

/// Check 4: each handle names a live object, no object has more handles than references, and a
/// resource limit has one reference more than its handles, its process row's.
fn check_handles(
    objects: &Objects,
    handles: &[(ProcessId, HandleTable)],
) -> Result<(), ObjectSnapshotError> {
    let mut uses: BTreeMap<ObjectId, u32> = BTreeMap::new();
    for (process, table) in handles {
        for (handle, object) in table.entries() {
            let fault = |fault| ObjectSnapshotError::Handle {
                process: *process,
                handle,
                fault,
            };
            let refs = objects.refs(object);
            if refs == 0 {
                return Err(fault(HandleFault::Stale));
            }
            let handles_so_far = uses.entry(object).or_insert(0);
            *handles_so_far = handles_so_far.saturating_add(1);
            if *handles_so_far > refs {
                return Err(fault(HandleFault::Overused));
            }
            // A resource limit's process row, which check 3 found names it, holds one more.
            let limit = matches!(objects.get(object), Ok(Object::ResourceLimit(_)));
            if limit && *handles_so_far == refs {
                return Err(fault(HandleFault::LimitRow));
            }
        }
    }
    Ok(())
}

/// Check 5: each hooked event, in order, is a live writable end its service's server made.
fn check_hooked(
    objects: &Objects,
    hooked: &[HookedEvent],
    kernel: &dyn KernelView,
) -> Result<(), ObjectSnapshotError> {
    let mut previous: Option<ObjectId> = None;
    for &hook in hooked {
        let fault = |fault| ObjectSnapshotError::Hook {
            hooked: hook,
            fault,
        };
        if previous.is_some_and(|before| hook.event <= before) {
            return Err(fault(HookFault::Order));
        }
        if !matches!(objects.get(hook.event), Ok(Object::Event(_))) {
            return Err(fault(HookFault::NotAnEvent));
        }
        if !kernel.has_service(hook.service) {
            return Err(fault(HookFault::UnknownService));
        }
        if !kernel.made_by(hook.service, hook.event) {
            return Err(fault(HookFault::NotMadeBy));
        }
        previous = Some(hook.event);
    }
    Ok(())
}

/// The object table and handle tables of a snapshot that passed checks 1 to 5, waiting for the
/// address spaces to be restored for check 6 ([`PendingObjects::finish`]). Only
/// [`Objects::restore`] (and [`Objects::save`], for its checks) makes one.
#[derive(Debug)]
pub struct PendingObjects {
    objects: Objects,
    handles: Vec<(ProcessId, HandleTable)>,
    hooked: Vec<HookedEvent>,
}

/// What a restore gives back once every check passed ([`PendingObjects::finish`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredObjects {
    /// The object table.
    pub objects: Objects,
    /// Each process's handle table, in process order.
    pub handles: Vec<(ProcessId, HandleTable)>,
    /// The hooked events, in event order: what the owning services must re-attach hooks to
    /// before the guest runs again
    /// ([`crate::dispatch::Dispatcher::unattached_event_hooks`]).
    pub hooked: Vec<HookedEvent>,
}

impl PendingObjects {
    /// The shared memory `object` names in the restored table, if it names one: what
    /// [`AddressSpace::restore`] is given to resolve its shared memory mappings with.
    #[must_use]
    pub fn shared_memory(&self, object: ObjectId) -> Option<SharedMemory> {
        match self.objects.get(object) {
            Ok(Object::SharedMemory(memory)) => Some(memory),
            _ => None,
        }
    }

    /// The hooked events the snapshot holds, in event order.
    #[must_use]
    pub fn hooked(&self) -> &[HookedEvent] {
        &self.hooked
    }

    /// Check 6 of the module docs against `spaces`, each process's restored address space in
    /// process order, and the restored tables.
    pub fn finish(
        self,
        spaces: &[(ProcessId, &AddressSpace)],
    ) -> Result<RestoredObjects, ObjectSnapshotError> {
        let mut previous: Option<ProcessId> = None;
        for &(process, _) in spaces {
            if previous.is_some_and(|before| process <= before) {
                return Err(ObjectSnapshotError::SpaceOrder { process });
            }
            previous = Some(process);
        }
        let mut expected = self.handles.iter().map(|(process, _)| *process).peekable();
        let mut given = spaces.iter().map(|(process, _)| *process).peekable();
        loop {
            match (expected.peek().copied(), given.peek().copied()) {
                (None, None) => break,
                (Some(wanted), Some(found)) if wanted == found => {
                    expected.next();
                    given.next();
                }
                (Some(process), None) => return Err(ObjectSnapshotError::MissingSpace { process }),
                (Some(wanted), Some(found)) if wanted < found => {
                    return Err(ObjectSnapshotError::MissingSpace { process: wanted });
                }
                (_, Some(process)) => return Err(ObjectSnapshotError::ExtraSpace { process }),
            }
        }
        let mut lends: BTreeMap<ProcessId, Vec<TransferMemory>> = BTreeMap::new();
        for (_, row) in self.objects.rows.iter() {
            if let Object::TransferMemory(transfer) = row.object {
                lends.entry(transfer.owner()).or_default().push(transfer);
            }
        }
        for &(process, space) in spaces {
            let own = lends.get(&process).map_or(&[][..], Vec::as_slice);
            space
                .check_lends(own)
                .map_err(|error| ObjectSnapshotError::Lends { process, error })?;
        }
        Ok(RestoredObjects {
            objects: self.objects,
            handles: self.handles,
            hooked: self.hooked,
        })
    }
}

impl Objects {
    /// The object table, `handles` (each process's handle table, in process order) and
    /// `hooked` (the hooked events, in event order) as a snapshot (module docs, "Format"),
    /// after checks 2 to 6 on what it is about to write against `kernel` and `spaces` (each
    /// process's address space, in process order): a failure is a bug in the kernel. A table
    /// with released transfer memories still to unlock is
    /// [`ObjectSnapshotError::PendingUnlocks`], checked first; then one with released shared
    /// memories whose pages are still to be given back is
    /// [`ObjectSnapshotError::PendingFrees`].
    pub fn save(
        &self,
        handles: &[(ProcessId, &HandleTable)],
        hooked: &[HookedEvent],
        kernel: &dyn KernelView,
        spaces: &[(ProcessId, &AddressSpace)],
    ) -> Result<Vec<u8>, ObjectSnapshotError> {
        if !self.unlocks.is_empty() {
            return Err(ObjectSnapshotError::PendingUnlocks {
                count: self.unlocks.len(),
            });
        }
        if !self.frees.is_empty() {
            return Err(ObjectSnapshotError::PendingFrees {
                count: self.frees.len(),
            });
        }
        let body = Body::of(self, handles, hooked);
        body.check(kernel)?.finish(spaces)?;
        Ok(encode(FORMAT, &body)?)
    }

    /// The tables a snapshot holds, after checks 1 to 5 of the module docs against `kernel`;
    /// check 6 waits for the address spaces ([`PendingObjects::finish`]).
    pub fn restore(
        bytes: &[u8],
        kernel: &dyn KernelView,
    ) -> Result<PendingObjects, ObjectSnapshotError> {
        let body: Body = decode(FORMAT, bytes)?;
        body.check(kernel)
    }
}
