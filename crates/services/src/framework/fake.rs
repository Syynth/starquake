//! [`FakeKernel`]: a [`Kernel`] for tests, with nothing behind it but tables in memory.
//!
//! It lets a service be driven through [`Server::handle`](super::Server::handle) without the
//! kernel crate's scheduler or a hypervisor. It is not a model of Horizon: events only count
//! their signals, sessions are bare ids, and guest memory is whatever ranges a test maps. Handles
//! are real: they come from the kernel crate's [`HandleTable`], so their values, the table's
//! capacity and its errors are Horizon's. Trace events go through a real
//! [`Recorder`](starquake_trace::Recorder), stamped with whatever caller the test set.
//!
//! [`FakeHost`] puts a [`FakeKernel`] and several servers together, so that a session one
//! service hands out for another ([`Kernel::connect`]) can be sent requests.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use starquake_kernel::handle::{Handle, HandleError, HandleTable, RefCounts};
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::memory::testing::standalone_shared_pages;
use starquake_kernel::memory::{
    SharedPages, SvcPermission, is_shared_memory_permission, shared_memory_size,
};
use starquake_kernel::result::{KernelError, ResultCode, SvcResult};
use starquake_trace::{
    Event, EventKind, GuestTicks, Recorder, ServiceId, ThreadId, VecSink, golden_trace,
};

use super::kernel::Kernel;
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
    handles: HandleTable,
    refs: RefCounts,
    minted: u32,
    events: BTreeMap<ObjectId, u32>,
    /// The shared memories made, with the permission the caller maps each with.
    shared: BTreeMap<ObjectId, (SharedPages, SvcPermission)>,
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
            handles: HandleTable::new(handle_table_size)?,
            refs: RefCounts::new(),
            minted: 0,
            events: BTreeMap::new(),
            shared: BTreeMap::new(),
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
    /// [`Kernel::create_session`] are on it.
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

    /// The pages of a shared memory [`Kernel::create_shared_memory`] made, and the permission
    /// the caller maps it with; `None` for any other id.
    #[must_use]
    pub fn shared_memory(&self, object: ObjectId) -> Option<(&SharedPages, SvcPermission)> {
        self.shared
            .get(&object)
            .map(|(pages, permission)| (pages, *permission))
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

    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        self.handles.get(handle).map_err(|e| kernel_error(&e))
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        let added = self.handles.add(&mut self.refs, object);
        added.map_err(|e| kernel_error(&e))
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        let event = self.new_object();
        self.events.insert(event, 0);
        Ok(event)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        let signals = self.events.get_mut(&event);
        let signals = signals.ok_or(KernelError::InvalidHandle)?;
        *signals = signals.saturating_add(1);
        Ok(())
    }

    /// The kernel's checks, in its order; the pages are host memory of their own, allocated
    /// at once.
    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        let size = shared_memory_size(size).ok_or(KernelError::InvalidSize)?;
        if !is_shared_memory_permission(permission) {
            return Err(KernelError::InvalidNewMemoryPermission);
        }
        let pages = standalone_shared_pages(size).ok_or(KernelError::OutOfMemory)?;
        let object = self.new_object();
        self.shared.insert(object, (pages.clone(), permission));
        Ok((object, pages))
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
}
