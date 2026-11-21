//! [`FakeKernel`]: a [`Kernel`] for tests, with nothing behind it but tables in memory.
//!
//! It lets a service be driven through [`Server::handle`](super::Server::handle) without the
//! kernel crate's scheduler or a hypervisor. It is not a model of Horizon: events only count
//! their signals, sessions are bare ids, and guest memory is whatever ranges a test maps. Handles
//! are real: they come from the kernel crate's [`HandleTable`], so their values, the table's
//! capacity and its errors are Horizon's. Trace events go through a real
//! [`Recorder`](starquake_trace::Recorder), stamped with whatever caller the test set.
//!
//! # Handle translation
//!
//! The kernel translates a request's copied handles before a server sees them
//! (`starquake_kernel::ipc::message`, `translate`): a copied [`Handle::CURRENT_THREAD`] or
//! [`Handle::CURRENT_PROCESS`] names the sending thread or process (Eden's
//! `KHandleTable::GetObjectForIpc`), and [`Kernel::resolve_handle`] answers it with that object.
//! The fake does the same with the objects a test gives it ([`FakeKernel::set_current`], the
//! kernel's own [`Current`]): a pseudo-handle names its object, and one whose object the test
//! did not give is [`KernelError::InvalidHandle`], as in the kernel. So a test can send libnx's
//! exact request, `CURRENT_PROCESS` included. The fake resolves a pseudo-handle wherever the
//! request has it; the kernel refuses a moved one before the request reaches a server (Eden's
//! move path takes no pseudo-handle), so a server never sees that case, and the fake does not
//! model the refusal.
//!
//! # Handles a reply does not name
//!
//! The fake delivers no reply to a kernel, so it does not take back the handles, or the
//! sessions, that a reply does not name. The real kernel does: when it settles a request, it
//! takes back every handle the handler placed in the sender's table that the delivered reply
//! does not name (`starquake_kernel`'s `dispatch/ipc/placed.rs`). So when the caller's table
//! fills up part-way through a reply, the handles and the session added before it stay in the
//! fake's table, where the kernel would remove them. Pinned by the tests
//! `handles_added_before_the_table_filled_stay_in_it` and
//! `a_session_opened_before_the_table_filled_stays_open` (`framework/tests/handles.rs`); the
//! kernel's behaviour is pinned by `tests/reply_rollback_on_the_kernel.rs`.
//!
//! # A restore's questions
//!
//! The fake answers a server's restore ([`RestoredKernel`]) from what it made: an event is the
//! service's when [`FakeKernel::set_current_service`] named that service as it was made (the
//! kernel notes the server handling the request), and a shared memory is one it made. It
//! differs from the kernel's [`Dispatcher`](starquake_kernel::dispatch::Dispatcher) in one way:
//! nothing the fake makes ever goes, so every id it made stays live, where the kernel's stay
//! live while a reference is held (a server keeps its creator's reference to what it made, so
//! the answers agree for those). The parity of the two is pinned by
//! `the_fake_and_the_kernel_answer_a_restore_alike` (`tests/am_on_the_kernel.rs`).
//!
//! [`FakeHost`] puts a [`FakeKernel`] and several servers together, so that a session one
//! service hands out for another ([`Kernel::connect`]) can be sent requests.
//!
//! The graphics server's calls are modelled as simply: a hooked event is two ids whose hook runs
//! when a test plays the guest's `SignalEvent` ([`FakeKernel::guest_signal`]); the GPU-accessible
//! attribute is whatever ranges a test marks ([`FakeKernel::mark_gpu_accessible`]); and a
//! device-share lock is a count per exact range ([`FakeKernel::device_locks`]), taken only on
//! mapped memory, whose token names [`FakeKernel::process`] (made with the kernel's
//! `test-support` constructor), and released only by a token naming that process and a range
//! that holds one. Ending the process does not make the fake refuse a token or a lock, as the
//! kernel does once the process has ended (`KernelError::TerminationRequested`). A transfer memory is a range of the caller's memory
//! a test lends ([`FakeKernel::lend`]); writes through it are bounded by the kernel's own check
//! (`starquake_kernel::memory::shared::span_within`) and reach only mapped memory. The fake
//! does not lock the lent pages or take the owner's permission away, as the kernel does. A
//! server keeps a reference of its own to a lent transfer memory as on the kernel
//! ([`Kernel::keep_transfer_memory`]): the fake counts the keeps per service (the one
//! [`FakeKernel::set_current_service`] names, as the kernel notes the server handling the
//! request) and transfer memory ([`FakeKernel::kept_transfer_memory`]), refuses a release that
//! has no keep behind it with [`KernelError::InvalidHandle`], and a transfer memory the test
//! ended while a keep was held ([`FakeKernel::end_transfer_memory`], the guest closing its
//! handle) stays until the last keep is released ([`FakeKernel::release_kept`], the kernel's
//! `Dispatcher::release_kept_transfer_memory` for a server's own thread). The kernel's rules are
//! pinned by `dispatch/hooks/tests/kept.rs` in the kernel crate, and the fake's by
//! `a_kept_transfer_memory_stays_after_the_guest_ends_it_until_the_last_release`.
//!
//! Clearing and ending are modelled as plainly. An event the fake made is signalled from a
//! signal until a clear ([`FakeKernel::is_signalled`]). Either end of a hooked event clears it,
//! as on the kernel: the writable end clears its readable end. Clearing any other id is the
//! kernel's [`KernelError::InvalidHandle`]. The fake makes no difference between servers, so it takes
//! every event it made as the caller's own, where the kernel lets only the server that made an
//! event clear it. Ending the process only counts the asks ([`FakeKernel::ends`]): the fake has
//! no process to end, and the reply comes before the end on the real kernel
//! (`starquake_kernel::dispatch`, the IPC arms' module docs).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use starquake_kernel::handle::{Handle, HandleError, HandleTable, RefCounts, Resolved};
use starquake_kernel::id::{ObjectId, ProcessId, SlabId};
use starquake_kernel::ipc::EventHook;
use starquake_kernel::ipc::message::Current;
use starquake_kernel::memory::shared::span_within;
use starquake_kernel::memory::testing::{device_share_token, standalone_shared_pages};
use starquake_kernel::memory::{
    DeviceShareToken, SharedMemory, SharedPages, SvcPermission, TransferMemory,
    is_shared_memory_permission, shared_memory_size,
};
use starquake_kernel::result::{KernelError, ResultCode, SvcResult};
use starquake_trace::{
    Event, EventKind, GuestTicks, Recorder, ServiceId, ThreadId, VecSink, golden_trace,
};

use super::kernel::Kernel;
use super::restored::RestoredKernel;
use super::server::{Handled, Server, ServerError};

/// The generation of every object id the fake mints.
const GENERATION: u32 = 1;

/// The kernel error a handle table error is to a guest. Errors with no guest-visible result
/// (the table's own bookkeeping) become [`KernelError::InvalidState`].
fn kernel_error(error: &HandleError) -> KernelError {
    let code = error.result().map(ResultCode::from_raw);
    code.and_then(KernelError::from_code)
        .unwrap_or(KernelError::InvalidState)
}

/// One mapped range of guest memory.
struct Region {
    base: u64,
    bytes: Vec<u8>,
}

impl Region {
    /// Where `len` bytes at `address` lie inside this region, if the whole extent does.
    fn range(&self, address: u64, len: usize) -> Option<core::ops::Range<usize>> {
        let start = usize::try_from(address.checked_sub(self.base)?).ok()?;
        let end = start.checked_add(len)?;
        (end <= self.bytes.len()).then_some(start..end)
    }
}

/// A [`Kernel`] made of in-memory tables. See the [module docs](self).
pub struct FakeKernel {
    process_id: u64,
    /// The objects the two pseudo-handles name for the caller.
    pseudo: Current,
    /// What every call that makes a kernel object answers instead, if a test set it.
    refusal: Option<KernelError>,
    /// How many calls that reach the refusal still succeed before it applies
    /// ([`FakeKernel::refuse_objects_after`]).
    allowance: u32,
    handles: HandleTable,
    refs: RefCounts,
    minted: u32,
    events: BTreeMap<ObjectId, u32>,
    /// The service whose server made each of the events: the current one
    /// ([`FakeKernel::set_current_service`]) when it was made.
    event_makers: BTreeMap<ObjectId, ServiceId>,
    /// The events made that are signalled now: a signal adds one, a clear takes it out.
    signalled: BTreeSet<ObjectId>,
    /// How often the caller asked to end its process.
    ends: u32,
    /// The hooked events made: each writable end's readable end and hook.
    hooked: BTreeMap<ObjectId, (ObjectId, Arc<dyn EventHook>)>,
    /// The ranges a test marked GPU-accessible, as `(address, size)`.
    gpu_accessible: Vec<(u64, u64)>,
    /// The device-share locks held, per exact range `(address, size)`.
    device_locks: BTreeMap<(u64, u64), u16>,
    /// The shared memories made, with the permission the caller maps each with.
    shared: BTreeMap<ObjectId, (SharedPages, SvcPermission)>,
    /// The transfer memories a test lent ([`FakeKernel::lend`]).
    transfers: BTreeMap<ObjectId, TransferMemory>,
    /// The references servers keep to lent transfer memories, by service and transfer memory,
    /// with how many ([`Kernel::keep_transfer_memory`]).
    kept: BTreeMap<(ServiceId, ObjectId), u32>,
    /// The lent transfer memories the test ended while a server kept one
    /// ([`FakeKernel::end_transfer_memory`]): each goes with its last keep.
    guest_closed: BTreeSet<ObjectId>,
    sessions: Vec<ObjectId>,
    regions: Vec<Region>,
    /// The services [`Kernel::connect`] knows.
    services: BTreeSet<ServiceId>,
    /// The service whose server is handling a request: new sessions are its.
    current: ServiceId,
    /// Which service's server each session is on.
    owners: BTreeMap<ObjectId, ServiceId>,
    /// Sessions [`Kernel::connect`] made that no server has been told of yet.
    connections: Vec<(ServiceId, ObjectId)>,
    recorder: Recorder,
    recorded: Arc<VecSink>,
    /// The stamp of the next recorded event.
    time: GuestTicks,
    core: u8,
    thread: ThreadId,
}

impl FakeKernel {
    /// A fake whose caller has a handle table of `handle_table_size` entries, with the meaning
    /// [`HandleTable::new`] gives the size (zero is the maximum). The process id is zero, no
    /// memory is mapped, and no object exists.
    pub fn new(handle_table_size: i32) -> Result<Self, HandleError> {
        let recorder = Recorder::new();
        let sink = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&sink));
        Ok(Self {
            process_id: 0,
            pseudo: Current::default(),
            refusal: None,
            allowance: 0,
            handles: HandleTable::new(handle_table_size)?,
            refs: RefCounts::new(),
            minted: 0,
            events: BTreeMap::new(),
            event_makers: BTreeMap::new(),
            signalled: BTreeSet::new(),
            ends: 0,
            hooked: BTreeMap::new(),
            gpu_accessible: Vec::new(),
            device_locks: BTreeMap::new(),
            shared: BTreeMap::new(),
            transfers: BTreeMap::new(),
            kept: BTreeMap::new(),
            guest_closed: BTreeSet::new(),
            sessions: Vec::new(),
            regions: Vec::new(),
            services: BTreeSet::new(),
            current: ServiceId(0),
            owners: BTreeMap::new(),
            connections: Vec::new(),
            recorder,
            recorded: sink,
            time: GuestTicks(0),
            core: 0,
            thread: ThreadId(0),
        })
    }

    /// Sets the guest thread and core the next requests come from, and the time: the stamp of
    /// every event recorded from now on.
    pub fn set_caller(&mut self, thread: ThreadId, core: u8, time: GuestTicks) {
        (self.thread, self.core, self.time) = (thread, core, time);
    }

    /// Every event recorded so far, in order.
    #[must_use]
    pub fn recorded(&self) -> Vec<Event> {
        self.recorded.events()
    }

    /// The golden-trace view of the events recorded so far.
    #[must_use]
    pub fn trace(&self) -> String {
        golden_trace(self.recorded.events().iter())
    }

    /// Makes `service` one [`Kernel::connect`] knows.
    pub fn add_service(&mut self, service: ServiceId) {
        self.services.insert(service);
    }

    /// Says which service's server handles the requests that follow: sessions made with
    /// [`Kernel::create_session`] are on it, and events made with [`Kernel::create_event`] and
    /// [`Kernel::create_hooked_event`] are its ([`RestoredKernel::is_service_readable_event`]).
    pub fn set_current_service(&mut self, service: ServiceId) {
        self.current = service;
    }

    /// The service whose server a session made by this fake is on.
    #[must_use]
    pub fn owner(&self, session: ObjectId) -> Option<ServiceId> {
        self.owners.get(&session).copied()
    }

    /// Takes the sessions [`Kernel::connect`] made since the last call, oldest first, each with
    /// the service it is on: the caller tells that service's server
    /// ([`Server::accept`](super::Server::accept)).
    pub fn take_connections(&mut self) -> Vec<(ServiceId, ObjectId)> {
        core::mem::take(&mut self.connections)
    }

    /// Sets the process id [`Kernel::process_id`] reports.
    pub fn set_process_id(&mut self, process_id: u64) {
        self.process_id = process_id;
    }

    /// Sets the objects [`Handle::CURRENT_THREAD`] and [`Handle::CURRENT_PROCESS`] name for the
    /// caller (module docs, "Handle translation"). Neither names anything until a test sets it.
    pub fn set_current(&mut self, current: Current) {
        self.pseudo = current;
    }

    /// Makes every call that creates a kernel object for a server ([`Kernel::create_event`],
    /// [`Kernel::create_shared_memory`]) fail with `refusal`, as the kernel's do when its object
    /// table is full ([`KernelError::OutOfResource`]); `None` lets them succeed again.
    pub fn refuse_objects(&mut self, refusal: Option<KernelError>) {
        self.refuse_objects_after(0, refusal);
    }

    /// As [`refuse_objects`](Self::refuse_objects), but the next `allowed` calls that would be
    /// refused succeed first, so a test can fail the Nth object a request makes. A call the
    /// kernel's own checks refuse before the refusal (a shared memory's size or permission)
    /// uses none of the allowance.
    pub fn refuse_objects_after(&mut self, allowed: u32, refusal: Option<KernelError>) {
        self.refusal = refusal;
        self.allowance = allowed;
    }

    /// The refusal a call that makes a kernel object meets now, if any; a call let through by
    /// the allowance uses one of it.
    fn refused(&mut self) -> Option<KernelError> {
        let refusal = self.refusal?;
        match self.allowance.checked_sub(1) {
            Some(left) => {
                self.allowance = left;
                None
            }
            None => Some(refusal),
        }
    }

    /// Mints a kernel object that is nothing in particular: something for a test to hold a
    /// handle to. Ids count up from index zero, so a run always mints the same ones.
    pub fn new_object(&mut self) -> ObjectId {
        let object = ObjectId::from_parts(self.minted, GENERATION);
        self.minted = self.minted.saturating_add(1);
        object
    }

    /// Number of handles in the caller's table.
    #[must_use]
    pub fn handle_count(&self) -> usize {
        self.handles.len()
    }

    /// How often `event` has been signalled, or `None` if it is not an event this fake created.
    #[must_use]
    pub fn signals(&self, event: ObjectId) -> Option<u32> {
        self.events.get(&event).copied()
    }

    /// Whether `event` is signalled now: signalled since its last clear. `None` if it is not an
    /// event this fake created.
    #[must_use]
    pub fn is_signalled(&self, event: ObjectId) -> Option<bool> {
        self.events
            .contains_key(&event)
            .then(|| self.signalled.contains(&event))
    }

    /// How often [`Kernel::end_process`] was called.
    #[must_use]
    pub fn ends(&self) -> u32 {
        self.ends
    }

    /// Plays the guest's `SignalEvent` on `writable`, the writable end of an event
    /// [`Kernel::create_hooked_event`] made: its readable end counts a signal
    /// ([`FakeKernel::signals`]), and then its hook runs, as the kernel runs it once its lock is
    /// released. [`KernelError::InvalidHandle`] for any other id, and nothing runs.
    pub fn guest_signal(&mut self, writable: ObjectId) -> SvcResult<()> {
        let (readable, hook) = self
            .hooked
            .get(&writable)
            .map(|(readable, hook)| (*readable, Arc::clone(hook)))
            .ok_or(KernelError::InvalidHandle)?;
        self.signal_event(readable)?;
        hook.signalled(writable);
        Ok(())
    }

    /// Marks the `size` bytes at `address` GPU-accessible, as the guest's `svc 0x80` would.
    pub fn mark_gpu_accessible(&mut self, address: u64, size: u64) {
        self.gpu_accessible.push((address, size));
    }

    /// The process the fake's device-share tokens name: the caller's, as the fake plays it
    /// (module docs).
    #[must_use]
    pub fn process(&self) -> ProcessId {
        ProcessId::from_parts(0, GENERATION)
    }

    /// How many device-share locks are held on exactly the `size` bytes at `address`.
    #[must_use]
    pub fn device_locks(&self, address: u64, size: u64) -> u16 {
        self.device_locks
            .get(&(address, size))
            .copied()
            .unwrap_or(0)
    }

    /// The pages of a shared memory [`Kernel::create_shared_memory`] made, and the permission
    /// the caller maps it with; `None` for any other id.
    #[must_use]
    pub fn shared_memory(&self, object: ObjectId) -> Option<(&SharedPages, SvcPermission)> {
        self.shared
            .get(&object)
            .map(|(pages, permission)| (pages, *permission))
    }

    /// Lends the `size` bytes at `address` of the caller's memory as a transfer memory, as the
    /// guest's `CreateTransferMemory` does, keeping `owner_permission` of them: a new object,
    /// for a test to give the caller a handle to and send in a request. The fake does not check
    /// that the range is mapped or page-aligned; reads and writes through it reach only mapped
    /// memory ([`FakeKernel::map`]).
    pub fn lend(&mut self, address: u64, size: u64, owner_permission: SvcPermission) -> ObjectId {
        let object = self.new_object();
        let lent = TransferMemory::new(self.process(), address, size, owner_permission);
        self.transfers.insert(object, lent.with_pages_locked());
        object
    }

    /// Ends the guest's hold on the transfer memory `object`, as closing its last handle does:
    /// unless a server keeps a reference to it ([`Kernel::keep_transfer_memory`]), it goes, and
    /// later calls on it are [`KernelError::InvalidHandle`]; if one does, it goes when the last
    /// keep is released ([`FakeKernel::release_kept`]). False if it named no transfer memory the
    /// guest still held.
    pub fn end_transfer_memory(&mut self, object: ObjectId) -> bool {
        if !self.transfers.contains_key(&object) || self.guest_closed.contains(&object) {
            return false;
        }
        if self.is_kept(object) {
            self.guest_closed.insert(object);
        } else {
            self.transfers.remove(&object);
        }
        true
    }

    /// How many references the server of `service` keeps to the transfer memory `object`
    /// ([`Kernel::keep_transfer_memory`]), as the kernel's `Dispatcher::kept_transfer_memory`.
    #[must_use]
    pub fn kept_transfer_memory(&self, service: ServiceId, object: ObjectId) -> u32 {
        self.kept.get(&(service, object)).copied().unwrap_or(0)
    }

    /// Drops one reference the server of `service` keeps to the transfer memory `object`, as
    /// the kernel's `Dispatcher::release_kept_transfer_memory` does on a server's own thread:
    /// [`KernelError::InvalidHandle`], with nothing changed, when it keeps none. When it was the
    /// last keep of a transfer memory the guest no longer holds
    /// ([`FakeKernel::end_transfer_memory`]), the transfer memory goes.
    pub fn release_kept(&mut self, service: ServiceId, object: ObjectId) -> SvcResult<()> {
        let count = self
            .kept
            .get_mut(&(service, object))
            .ok_or(KernelError::InvalidHandle)?;
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.kept.remove(&(service, object));
        }
        if !self.is_kept(object) && self.guest_closed.remove(&object) {
            self.transfers.remove(&object);
        }
        Ok(())
    }

    /// Whether any server keeps a reference to `object`.
    fn is_kept(&self, object: ObjectId) -> bool {
        self.kept.keys().any(|(_, kept)| *kept == object)
    }

    /// The caller's address of `len` bytes at `offset` into the transfer memory `object`, as
    /// the kernel finds it: [`KernelError::InvalidHandle`] unless `object` is one
    /// [`FakeKernel::lend`] made, and [`KernelError::InvalidCurrentMemory`] unless the whole
    /// extent lies inside it.
    pub fn lent_range(&self, object: ObjectId, offset: u64, len: usize) -> SvcResult<u64> {
        let lent = self.transfer_memory(object)?;
        span_within(lent.address(), lent.size(), offset, len)
            .ok_or(KernelError::InvalidCurrentMemory)
    }

    /// The sessions [`Kernel::create_session`] has created, oldest first.
    #[must_use]
    pub fn sessions(&self) -> &[ObjectId] {
        &self.sessions
    }

    /// Maps `bytes` at `address` in the caller's memory. A read or write has to lie inside one
    /// mapped range; ranges that touch are not joined. Where ranges overlap, the one mapped
    /// first answers.
    pub fn map(&mut self, address: u64, bytes: Vec<u8>) {
        self.regions.push(Region {
            base: address,
            bytes,
        });
    }

    /// The `len` bytes at `address` in the caller's memory, or `None` unless one mapped range
    /// holds them all.
    #[must_use]
    pub fn memory(&self, address: u64, len: usize) -> Option<&[u8]> {
        self.regions.iter().find_map(|region| {
            let range = region.range(address, len)?;
            region.bytes.get(range)
        })
    }
}

impl Kernel for FakeKernel {
    fn process_id(&self) -> u64 {
        self.process_id
    }

    /// A pseudo-handle names the object [`FakeKernel::set_current`] gave it, or is
    /// [`KernelError::InvalidHandle`] (module docs, "Handle translation"); any other handle is
    /// looked up in the caller's table.
    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        let resolved = self.handles.resolve(handle).map_err(|e| kernel_error(&e))?;
        let object = match resolved {
            Resolved::CurrentThread => self.pseudo.thread,
            Resolved::CurrentProcess => self.pseudo.process,
            Resolved::Object(object) => Some(object),
        };
        object.ok_or(KernelError::InvalidHandle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        let added = self.handles.add(&mut self.refs, object);
        added.map_err(|e| kernel_error(&e))
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        if let Some(refusal) = self.refused() {
            return Err(refusal);
        }
        let event = self.new_object();
        self.events.insert(event, 0);
        self.event_makers.insert(event, self.current);
        Ok(event)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        let signals = self.events.get_mut(&event);
        let signals = signals.ok_or(KernelError::InvalidHandle)?;
        *signals = signals.saturating_add(1);
        self.signalled.insert(event);
        Ok(())
    }

    /// Clears an event this fake made: an id [`Kernel::create_event`] made, or either end of
    /// one [`Kernel::create_hooked_event`] made (the writable end clears its readable end, as
    /// the kernel's `IpcContext::clear_event` does). [`KernelError::InvalidHandle`] for any
    /// other id, and nothing changes.
    fn clear_event(&mut self, event: ObjectId) -> SvcResult<()> {
        let readable = match self.hooked.get(&event) {
            Some((readable, _)) => *readable,
            None => event,
        };
        if !self.events.contains_key(&readable) {
            return Err(KernelError::InvalidHandle);
        }
        self.signalled.remove(&readable);
        Ok(())
    }

    /// Counts the ask ([`FakeKernel::ends`]).
    fn end_process(&mut self) -> SvcResult<()> {
        self.ends = self.ends.saturating_add(1);
        Ok(())
    }

    /// The kernel's checks, in its order; the pages are host memory of their own, allocated
    /// at once, so more than
    /// [`SHARED_CAPACITY`](starquake_kernel::memory::testing::SHARED_CAPACITY) bytes is [`KernelError::OutOfMemory`]
    /// ([`standalone_shared_pages`]).
    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        let size = shared_memory_size(size).ok_or(KernelError::InvalidSize)?;
        if !is_shared_memory_permission(permission) {
            return Err(KernelError::InvalidNewMemoryPermission);
        }
        if let Some(refusal) = self.refused() {
            return Err(refusal);
        }
        let pages = standalone_shared_pages(size).ok_or(KernelError::OutOfMemory)?;
        let object = self.new_object();
        self.shared.insert(object, (pages.clone(), permission));
        Ok((object, pages))
    }

    /// Two fresh ids, the writable end and then the readable end; the readable end counts
    /// signals as [`Kernel::create_event`]'s events do.
    fn create_hooked_event(&mut self, hook: Arc<dyn EventHook>) -> SvcResult<(ObjectId, ObjectId)> {
        let writable = self.new_object();
        let readable = self.create_event()?;
        self.hooked.insert(writable, (readable, hook));
        Ok((writable, readable))
    }

    /// True when one range a test marked holds the whole of the `size` bytes at `address`
    /// (ranges that touch are not joined); false for an empty range or one that wraps.
    fn is_gpu_accessible(&self, address: u64, size: u64) -> SvcResult<bool> {
        let Some(end) = address.checked_add(size).filter(|_| size > 0) else {
            return Ok(false);
        };
        Ok(self.gpu_accessible.iter().any(|(start, len)| {
            start
                .checked_add(*len)
                .is_some_and(|marked_end| *start <= address && end <= marked_end)
        }))
    }

    /// [`KernelError::InvalidCurrentMemory`] unless one mapped range holds the whole of the
    /// `size` bytes at `address`, and [`KernelError::OutOfResource`] when the range already
    /// holds `u16::MAX` locks, as the kernel answers. The token names [`FakeKernel::process`].
    fn lock_device_share(&mut self, address: u64, size: u64) -> SvcResult<DeviceShareToken> {
        let len = usize::try_from(size).map_err(|_| KernelError::InvalidCurrentMemory)?;
        if size == 0 || self.memory(address, len).is_none() {
            return Err(KernelError::InvalidCurrentMemory);
        }
        let count = self.device_locks.entry((address, size)).or_insert(0);
        *count = count.checked_add(1).ok_or(KernelError::OutOfResource)?;
        Ok(device_share_token(self.process(), address, size))
    }

    /// [`KernelError::InvalidCurrentMemory`] unless the token names [`FakeKernel::process`] and
    /// a lock is held on exactly its range, as the kernel answers a server. The fake has no
    /// process to end, so it never refuses a token as the kernel does once its process has
    /// ended (module docs).
    fn unlock_device_share(&mut self, token: DeviceShareToken) -> SvcResult<()> {
        if token.process() != self.process() {
            return Err(KernelError::InvalidCurrentMemory);
        }
        let range = (token.address(), token.size());
        let count = self
            .device_locks
            .get_mut(&range)
            .ok_or(KernelError::InvalidCurrentMemory)?;
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.device_locks.remove(&range);
        }
        Ok(())
    }

    /// A transfer memory [`FakeKernel::lend`] made and [`FakeKernel::end_transfer_memory`] has
    /// not ended; [`KernelError::InvalidHandle`] for any other id.
    fn transfer_memory(&self, object: ObjectId) -> SvcResult<TransferMemory> {
        self.transfers
            .get(&object)
            .copied()
            .ok_or(KernelError::InvalidHandle)
    }

    /// Writes where [`FakeKernel::lent_range`] finds `data`'s extent, through
    /// [`Kernel::write_memory`]: [`KernelError::InvalidCurrentMemory`] also when that memory is
    /// not mapped. Nothing is written on an error.
    fn write_transfer_memory(
        &mut self,
        object: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        let address = self.lent_range(object, offset, data.len())?;
        self.write_memory(address, data)
    }

    /// Counts one keep of `object` for the current service ([`FakeKernel::set_current_service`]);
    /// [`KernelError::InvalidHandle`], with nothing kept, unless it is a live lent transfer
    /// memory.
    fn keep_transfer_memory(&mut self, object: ObjectId) -> SvcResult<TransferMemory> {
        let lent = self.transfer_memory(object)?;
        let count = self.kept.entry((self.current, object)).or_insert(0);
        *count = count.saturating_add(1);
        Ok(lent)
    }

    /// [`FakeKernel::release_kept`] for the current service.
    fn release_kept_transfer_memory(&mut self, object: ObjectId) -> SvcResult<()> {
        self.release_kept(self.current, object)
    }

    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        // Peek at the id: it is only minted once the handle is in, so a full table leaves no
        // trace.
        let session = ObjectId::from_parts(self.minted, GENERATION);
        let handle = self.insert_handle(session)?;
        self.minted = self.minted.saturating_add(1);
        self.sessions.push(session);
        self.owners.insert(session, self.current);
        Ok((session, handle))
    }

    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        if !self.services.contains(&service) {
            return Err(KernelError::NotFound);
        }
        // As `create_session`: the id is minted only once the handle is in.
        let session = ObjectId::from_parts(self.minted, GENERATION);
        let handle = self.insert_handle(session)?;
        self.minted = self.minted.saturating_add(1);
        self.owners.insert(session, service);
        self.connections.push((service, session));
        Ok((session, handle))
    }

    fn record(&mut self, kind: EventKind) {
        self.recorder
            .record(self.time, self.core, self.thread, kind);
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        let bytes = self.memory(address, out.len());
        out.copy_from_slice(bytes.ok_or(KernelError::InvalidCurrentMemory)?);
        Ok(())
    }

    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        let found = self.regions.iter_mut().find_map(|region| {
            let range = region.range(address, data.len())?;
            region.bytes.get_mut(range)
        });
        found
            .ok_or(KernelError::InvalidCurrentMemory)?
            .copy_from_slice(data);
        Ok(())
    }
}

/// The objects this fake made, as a restore asks about them (module docs, "A restore's
/// questions").
impl RestoredKernel for FakeKernel {
    /// True for an id [`Kernel::create_event`] made, or the readable end
    /// [`Kernel::create_hooked_event`] made, while `service` was the current service.
    fn is_service_readable_event(&self, service: ServiceId, event: ObjectId) -> bool {
        self.events.contains_key(&event) && self.event_makers.get(&event) == Some(&service)
    }

    /// A shared memory [`Kernel::create_shared_memory`] made, with its pages and permission.
    fn shared_memory(&self, object: ObjectId) -> Option<SharedMemory> {
        let (pages, permission) = self.shared.get(&object)?;
        SharedMemory::new(pages.backing(), *permission).ok()
    }
}

/// Why a [`FakeHost`] could not pass a message on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostError {
    /// The handle names no session this host made.
    #[error("handle {0:#x} names no session of this host")]
    UnknownSession(u32),
    /// No server was added for the service.
    #[error("no server for service {0:?}")]
    UnknownService(ServiceId),
    /// The kernel refused.
    #[error("kernel: {0}")]
    Kernel(#[from] KernelError),
    /// The server failed.
    #[error(transparent)]
    Server(#[from] ServerError),
}

/// A [`FakeKernel`] and the servers of several services: enough of a system to get a service
/// from another one and talk to it. It stands in for the kernel's session code: it routes a
/// message to the server its session is on, and tells a server of the sessions made on it.
///
/// It does not queue: a message is handled at once, in the caller. Tests of the request queue
/// drive a [`Server`] directly.
pub struct FakeHost {
    /// The kernel every server of the host is given.
    pub kernel: FakeKernel,
    servers: BTreeMap<ServiceId, Server>,
}

impl FakeHost {
    /// A host with no servers.
    #[must_use]
    pub fn new(kernel: FakeKernel) -> Self {
        Self {
            kernel,
            servers: BTreeMap::new(),
        }
    }

    /// Adds a server, under its [`Server::service_id`]. A server added under a number already
    /// taken replaces the earlier one.
    pub fn add(&mut self, server: Server) {
        self.kernel.add_service(server.service_id());
        self.servers.insert(server.service_id(), server);
    }

    /// The server of a service.
    #[must_use]
    pub fn server(&self, service: ServiceId) -> Option<&Server> {
        self.servers.get(&service)
    }

    /// The server of a service, to change it.
    pub fn server_mut(&mut self, service: ServiceId) -> Option<&mut Server> {
        self.servers.get_mut(&service)
    }

    /// Connects the caller to a service, as connecting to its port would: a new session on its
    /// server, opened with the service's root interface. Returns the caller's handle.
    pub fn connect(&mut self, service: ServiceId) -> Result<Handle, HostError> {
        let (_, handle) = self.kernel.connect(service)?;
        self.accept_connections()?;
        Ok(handle)
    }

    /// Sends the message in `buffer` on the session `handle` names, to whichever server the
    /// session is on, and then opens the sessions the reply handed out on other services.
    pub fn send(&mut self, handle: Handle, buffer: &mut [u8]) -> Result<Handled, HostError> {
        let unknown = HostError::UnknownSession(handle.raw());
        let session = self.kernel.resolve_handle(handle);
        let session = session.map_err(|_| unknown.clone())?;
        let service = self.kernel.owner(session).ok_or(unknown)?;
        let server = self.servers.get_mut(&service);
        let server = server.ok_or(HostError::UnknownService(service))?;
        self.kernel.set_current_service(service);
        let handled = server.handle(&mut self.kernel, session, buffer)?;
        self.accept_connections()?;
        Ok(handled)
    }

    /// Tells each server of the sessions made on it since the last call.
    fn accept_connections(&mut self) -> Result<(), HostError> {
        for (service, session) in self.kernel.take_connections() {
            let server = self.servers.get_mut(&service);
            let server = server.ok_or(HostError::UnknownService(service))?;
            server.accept(session)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use starquake_kernel::handle::MAX_TABLE_SIZE;

    use super::*;

    #[test]
    fn handles_come_from_a_real_table_and_run_out_at_its_size() {
        let mut kernel = FakeKernel::new(2).unwrap();
        let (a, b) = (kernel.new_object(), kernel.new_object());
        assert_ne!(a, b);
        let first = kernel.insert_handle(a).unwrap();
        let second = kernel.insert_handle(b).unwrap();
        assert_eq!(kernel.resolve_handle(first), Ok(a));
        assert_eq!(kernel.resolve_handle(second), Ok(b));
        assert_eq!(kernel.handle_count(), 2);
        // Exactly at the limit: the third does not fit.
        assert_eq!(kernel.insert_handle(a), Err(KernelError::OutOfHandles));
        assert_eq!(
            kernel.resolve_handle(Handle::INVALID),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(
            kernel.resolve_handle(Handle::CURRENT_PROCESS),
            Err(KernelError::InvalidHandle)
        );
    }

    #[test]
    fn the_table_size_means_what_the_kernel_crate_says() {
        assert_eq!(
            FakeKernel::new(0).unwrap().handles.capacity(),
            usize::from(MAX_TABLE_SIZE)
        );
        let too_large = i32::from(MAX_TABLE_SIZE) + 1;
        assert!(matches!(
            FakeKernel::new(too_large),
            Err(HandleError::TableTooLarge { requested }) if requested == too_large
        ));
    }

    #[test]
    fn events_count_signals_and_only_events_can_be_signalled() {
        let mut kernel = FakeKernel::new(0).unwrap();
        let object = kernel.new_object();
        let event = kernel.create_event().unwrap();
        assert_eq!(kernel.signals(event), Some(0));
        kernel.signal_event(event).unwrap();
        kernel.signal_event(event).unwrap();
        assert_eq!(kernel.signals(event), Some(2));
        assert_eq!(kernel.signals(object), None);
        assert_eq!(kernel.signal_event(object), Err(KernelError::InvalidHandle));
    }

    /// A hook that does nothing.
    struct Deaf;

    impl EventHook for Deaf {
        fn signalled(&self, _: ObjectId) {}
    }

    #[test]
    fn either_end_of_a_hooked_event_clears_it_and_any_other_id_is_an_invalid_handle() {
        let mut kernel = FakeKernel::new(0).unwrap();
        let (writable, readable) = kernel.create_hooked_event(Arc::new(Deaf)).unwrap();
        for end in [writable, readable] {
            kernel.guest_signal(writable).unwrap();
            assert_eq!(kernel.is_signalled(readable), Some(true), "{end:?}");
            assert_eq!(kernel.clear_event(end), Ok(()), "{end:?}");
            assert_eq!(kernel.is_signalled(readable), Some(false), "{end:?}");
        }
        // An id that is no event stays refused, and the event is not touched.
        kernel.guest_signal(writable).unwrap();
        let object = kernel.new_object();
        assert_eq!(kernel.clear_event(object), Err(KernelError::InvalidHandle));
        assert_eq!(kernel.is_signalled(readable), Some(true));
    }

    #[test]
    fn a_session_comes_with_its_handle_or_not_at_all() {
        let mut kernel = FakeKernel::new(1).unwrap();
        let (session, handle) = kernel.create_session().unwrap();
        assert_eq!(kernel.resolve_handle(handle), Ok(session));
        assert_eq!(kernel.sessions(), [session]);
        // The table is full: no session, no handle, and the next object id is not used up.
        assert_eq!(kernel.create_session(), Err(KernelError::OutOfHandles));
        assert_eq!(kernel.sessions(), [session]);
        assert_eq!(kernel.handle_count(), 1);
        assert_eq!(kernel.new_object().index(), session.index() + 1);
    }

    #[test]
    fn a_connection_needs_a_known_service_and_room_for_its_handle() {
        const SERVICE: ServiceId = ServiceId(4);
        let mut kernel = FakeKernel::new(1).unwrap();
        assert_eq!(kernel.connect(SERVICE), Err(KernelError::NotFound));
        kernel.add_service(SERVICE);
        let (session, handle) = kernel.connect(SERVICE).unwrap();
        assert_eq!(kernel.resolve_handle(handle), Ok(session));
        assert_eq!(kernel.owner(session), Some(SERVICE));
        // The table is full: no session, no handle, nothing for a server to accept.
        assert_eq!(kernel.connect(SERVICE), Err(KernelError::OutOfHandles));
        assert_eq!(kernel.take_connections(), [(SERVICE, session)]);
        assert_eq!(kernel.take_connections(), []);
        assert_eq!(kernel.new_object().index(), session.index() + 1);
    }

    #[test]
    fn a_new_session_is_on_the_server_that_is_handling_a_request() {
        let mut kernel = FakeKernel::new(0).unwrap();
        let (first, _) = kernel.create_session().unwrap();
        kernel.set_current_service(ServiceId(9));
        let (second, _) = kernel.create_session().unwrap();
        assert_eq!(kernel.owner(first), Some(ServiceId(0)));
        assert_eq!(kernel.owner(second), Some(ServiceId(9)));
        let object = kernel.new_object();
        assert_eq!(kernel.owner(object), None);
    }

    #[test]
    fn events_are_stamped_with_the_caller_that_was_set() {
        let note = |value| EventKind::ServiceNote {
            service: ServiceId(1),
            code: 2,
            value,
        };
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.record(note(10));
        kernel.set_caller(ThreadId(7), 3, GuestTicks(99));
        kernel.record(note(11));
        let stamps = |e: &Event| (e.seq, e.time, e.core, e.thread, e.kind);
        assert_eq!(
            kernel.recorded().iter().map(stamps).collect::<Vec<_>>(),
            [
                (0, GuestTicks(0), 0, ThreadId(0), note(10)),
                (1, GuestTicks(99), 3, ThreadId(7), note(11)),
            ]
        );
        assert_eq!(
            kernel.trace(),
            format!("thread 0\n  {}\nthread 7\n  {}\n", note(10), note(11))
        );
    }

    #[test]
    fn the_process_id_is_what_was_set() {
        let mut kernel = FakeKernel::new(0).unwrap();
        assert_eq!(kernel.process_id(), 0);
        kernel.set_process_id(0x1234_5678_9ABC);
        assert_eq!(kernel.process_id(), 0x1234_5678_9ABC);
    }

    #[test]
    fn a_pseudo_handle_names_the_object_set_for_it_and_nothing_until_then() {
        let mut kernel = FakeKernel::new(0).unwrap();
        let (thread, process, other) = (
            kernel.new_object(),
            kernel.new_object(),
            kernel.new_object(),
        );
        let held = kernel.insert_handle(other).unwrap();
        let unset = Err(KernelError::InvalidHandle);
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_THREAD), unset);
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_PROCESS), unset);
        // Each names its own object, so swapping the two shows.
        kernel.set_current(Current {
            thread: Some(thread),
            process: Some(process),
        });
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_THREAD), Ok(thread));
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_PROCESS), Ok(process));
        // Only the one that was given: the other is the kernel's error.
        kernel.set_current(Current {
            thread: None,
            process: Some(process),
        });
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_THREAD), unset);
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_PROCESS), Ok(process));
        kernel.set_current(Current {
            thread: Some(thread),
            process: None,
        });
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_THREAD), Ok(thread));
        assert_eq!(kernel.resolve_handle(Handle::CURRENT_PROCESS), unset);
        // An ordinary handle is still the table's, and a pseudo-handle adds nothing to it.
        assert_eq!(kernel.resolve_handle(held), Ok(other));
        assert_eq!(kernel.handle_count(), 1);
    }

    #[test]
    fn a_refusal_fails_every_object_a_server_makes_until_it_is_lifted() {
        const SIZE: u64 = 0x1000;
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.refuse_objects(Some(KernelError::OutOfResource));
        let refused = Err(KernelError::OutOfResource);
        assert_eq!(kernel.create_event(), refused);
        assert_eq!(
            kernel
                .create_shared_memory(SIZE, SvcPermission::Read)
                .map(|(object, _)| object),
            refused
        );
        // The kernel's own checks still come first: the size, then the permission.
        assert_eq!(
            kernel
                .create_shared_memory(0, SvcPermission::Read)
                .map(|(object, _)| object),
            Err(KernelError::InvalidSize)
        );
        assert!(!is_shared_memory_permission(SvcPermission::ReadExecute));
        assert_eq!(
            kernel
                .create_shared_memory(SIZE, SvcPermission::ReadExecute)
                .map(|(object, _)| object),
            Err(KernelError::InvalidNewMemoryPermission)
        );
        // Nothing was made, so the next object is the first.
        kernel.refuse_objects(None);
        assert_eq!(
            kernel.create_event(),
            Ok(ObjectId::from_parts(0, GENERATION))
        );
        let (memory, _) = kernel
            .create_shared_memory(SIZE, SvcPermission::Read)
            .unwrap();
        assert_eq!(memory, ObjectId::from_parts(1, GENERATION));
    }

    #[test]
    fn a_refusal_after_an_allowance_lets_that_many_objects_be_made_first() {
        const SIZE: u64 = 0x1000;
        let mut kernel = FakeKernel::new(0).unwrap();
        let refused = Err(KernelError::OutOfResource);
        kernel.refuse_objects_after(2, Some(KernelError::OutOfResource));
        assert_eq!(
            kernel.create_event(),
            Ok(ObjectId::from_parts(0, GENERATION))
        );
        // A call the kernel's own checks refuse uses none of the allowance.
        assert_eq!(
            kernel
                .create_shared_memory(0, SvcPermission::Read)
                .map(|(object, _)| object),
            Err(KernelError::InvalidSize)
        );
        assert_eq!(
            kernel
                .create_shared_memory(SIZE, SvcPermission::Read)
                .map(|(object, _)| object),
            Ok(ObjectId::from_parts(1, GENERATION))
        );
        // The allowance is used: every later call is refused, and nothing is made.
        assert_eq!(kernel.create_event(), refused);
        assert_eq!(
            kernel
                .create_shared_memory(SIZE, SvcPermission::Read)
                .map(|(object, _)| object),
            refused
        );
        assert_eq!(kernel.create_event(), refused);
        // `refuse_objects` sets no allowance, even after one was set.
        kernel.refuse_objects_after(1, Some(KernelError::OutOfResource));
        kernel.refuse_objects(Some(KernelError::OutOfResource));
        assert_eq!(kernel.create_event(), refused);
        // No refusal: every call succeeds, whatever the allowance.
        kernel.refuse_objects_after(0, None);
        assert_eq!(
            kernel.create_event(),
            Ok(ObjectId::from_parts(2, GENERATION))
        );
        assert_eq!(
            kernel.create_event(),
            Ok(ObjectId::from_parts(3, GENERATION))
        );
    }

    #[test]
    fn memory_access_needs_the_whole_extent_inside_one_range() {
        const BASE: u64 = 0x1000;
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(BASE, vec![1, 2, 3, 4]);
        let mut out = [0; 4];
        kernel.read_memory(BASE, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        kernel.write_memory(BASE + 2, &[9, 8]).unwrap();
        assert_eq!(kernel.memory(BASE, 4), Some(&[1, 2, 9, 8][..]));

        let fault = Err(KernelError::InvalidCurrentMemory);
        // One byte past the end, one byte before the start, and a range that starts inside.
        assert_eq!(kernel.read_memory(BASE + 1, &mut out), fault);
        assert_eq!(kernel.read_memory(BASE - 1, &mut out), fault);
        assert_eq!(kernel.write_memory(BASE + 3, &[7, 7]), fault);
        assert_eq!(kernel.write_memory(BASE - 1, &[7]), fault);
        // A failed write changes nothing, and an empty access at the end is inside.
        assert_eq!(kernel.memory(BASE, 4), Some(&[1, 2, 9, 8][..]));
        assert_eq!(kernel.read_memory(BASE + 4, &mut []), Ok(()));
        assert_eq!(kernel.read_memory(BASE + 5, &mut []), fault);
    }

    #[test]
    fn an_extent_that_wraps_the_address_space_is_a_fault() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(u64::MAX - 1, vec![5, 6]);
        let mut out = [0; 2];
        // The last two bytes of the address space are mapped and readable...
        kernel.read_memory(u64::MAX - 1, &mut out).unwrap();
        assert_eq!(out, [5, 6]);
        // ...but nothing reaches past them, from the last byte or from address zero.
        let fault = Err(KernelError::InvalidCurrentMemory);
        assert_eq!(kernel.read_memory(u64::MAX, &mut out), fault);
        assert_eq!(kernel.write_memory(u64::MAX, &[1, 2]), fault);
        assert_eq!(kernel.read_memory(0, &mut out), fault);
    }

    #[test]
    fn ranges_that_touch_are_not_joined() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(0x1000, vec![1, 2]);
        kernel.map(0x1002, vec![3, 4]);
        let mut out = [0; 2];
        assert_eq!(
            kernel.read_memory(0x1001, &mut out),
            Err(KernelError::InvalidCurrentMemory)
        );
        kernel.read_memory(0x1002, &mut out).unwrap();
        assert_eq!(out, [3, 4]);
    }

    #[test]
    fn a_lent_transfer_memory_is_written_within_its_bounds_and_its_neighbour_is_untouched() {
        const BASE: u64 = 0x4000;
        const SIZE: u64 = 0x10;
        let mut kernel = FakeKernel::new(0).unwrap();
        // The lent range and the byte after it, which starts exactly at its end bound.
        kernel.map(BASE, vec![0; 0x11]);
        let lent = kernel.lend(BASE, SIZE, SvcPermission::Read);
        let found = kernel.transfer_memory(lent).unwrap();
        assert_eq!(
            (found.owner(), found.address(), found.size()),
            (kernel.process(), BASE, SIZE)
        );
        assert_eq!(found.owner_permission(), SvcPermission::Read);
        assert!(found.is_locked());
        // Up to exactly the end is written; one byte past it is refused and writes nothing.
        kernel
            .write_transfer_memory(lent, SIZE - 2, &[7, 8])
            .unwrap();
        assert_eq!(
            kernel.write_transfer_memory(lent, SIZE - 1, &[9, 9]),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(
            kernel.write_transfer_memory(lent, u64::MAX, &[1]),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(kernel.memory(BASE + SIZE - 2, 3), Some(&[7, 8, 0][..]));
        assert_eq!(kernel.lent_range(lent, 3, 1), Ok(BASE + 3));
    }

    #[test]
    fn only_a_live_transfer_memory_is_one() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(0x1000, vec![0; 4]);
        let other = kernel.new_object();
        let lent = kernel.lend(0x1000, 4, SvcPermission::None);
        assert_eq!(
            kernel.transfer_memory(other),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(
            kernel.write_transfer_memory(other, 0, &[1]),
            Err(KernelError::InvalidHandle)
        );
        assert!(kernel.end_transfer_memory(lent));
        assert!(!kernel.end_transfer_memory(lent));
        assert_eq!(
            kernel.transfer_memory(lent),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(
            kernel.write_transfer_memory(lent, 0, &[1]),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(kernel.memory(0x1000, 4), Some(&[0; 4][..]));
    }

    #[test]
    fn a_kept_transfer_memory_stays_after_the_guest_ends_it_until_the_last_release() {
        const SERVICE: ServiceId = ServiceId(0x31);
        const OTHER: ServiceId = ServiceId(0x32);
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(0x1000, vec![0; 4]);
        let lent = kernel.lend(0x1000, 4, SvcPermission::None);
        let event = kernel.create_event().unwrap();
        kernel.set_current_service(SERVICE);
        // Only a live lent transfer memory is kept.
        assert_eq!(
            kernel.keep_transfer_memory(event).map(|_| ()),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(kernel.kept_transfer_memory(SERVICE, event), 0);
        let found = kernel.keep_transfer_memory(lent).unwrap();
        assert_eq!((found.address(), found.size()), (0x1000, 4));
        kernel.keep_transfer_memory(lent).unwrap();
        assert_eq!(kernel.kept_transfer_memory(SERVICE, lent), 2);
        assert_eq!(kernel.kept_transfer_memory(OTHER, lent), 0);
        // Another service's release has nothing behind it.
        assert_eq!(
            kernel.release_kept(OTHER, lent),
            Err(KernelError::InvalidHandle)
        );
        // The guest ends it: it stays while a keep is held.
        assert!(kernel.end_transfer_memory(lent));
        assert!(!kernel.end_transfer_memory(lent));
        kernel.write_transfer_memory(lent, 0, &[5]).unwrap();
        kernel.release_kept_transfer_memory(lent).unwrap();
        assert_eq!(kernel.kept_transfer_memory(SERVICE, lent), 1);
        kernel.write_transfer_memory(lent, 1, &[6]).unwrap();
        // The last keep goes, and the transfer memory with it.
        kernel.release_kept(SERVICE, lent).unwrap();
        assert_eq!(kernel.kept_transfer_memory(SERVICE, lent), 0);
        assert_eq!(
            kernel.transfer_memory(lent),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(
            kernel.release_kept(SERVICE, lent),
            Err(KernelError::InvalidHandle)
        );
        assert_eq!(kernel.memory(0x1000, 4), Some(&[5, 6, 0, 0][..]));
    }

    #[test]
    fn releasing_a_keep_before_the_guest_ends_it_leaves_it_live() {
        const SERVICE: ServiceId = ServiceId(0x31);
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(0x1000, vec![0; 4]);
        let lent = kernel.lend(0x1000, 4, SvcPermission::None);
        kernel.set_current_service(SERVICE);
        kernel.keep_transfer_memory(lent).unwrap();
        kernel.release_kept_transfer_memory(lent).unwrap();
        assert!(kernel.transfer_memory(lent).is_ok());
        assert_eq!(
            kernel.release_kept_transfer_memory(lent),
            Err(KernelError::InvalidHandle)
        );
        // Ending it now makes it go at once.
        assert!(kernel.end_transfer_memory(lent));
        assert_eq!(
            kernel.transfer_memory(lent),
            Err(KernelError::InvalidHandle)
        );
    }
}
