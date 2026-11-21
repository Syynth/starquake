//! The page's restore against the kernel's own dispatcher, not the [`Holding`] double: the root
//! server with the page registered on a real dispatcher, the title's two transfer memories lent
//! by its own `CreateTransferMemory`, 204 sent through the kernel, and the restore's checks run
//! against what the kernel then holds (owner's ruling of 2026-11-17: hooks are re-attached by
//! their owner in its restore step, checked against the restored events). Each refusal is also
//! run on [`Holding`] holding the same objects, which must answer alike. The root's own
//! references to the page's memories are followed here end to end too: the title closes its
//! handles and the page still answers through the kernel, and the root's last session closing
//! lets them go (issue 499).

use starquake_cpu::GuestRegs;
use starquake_kernel::dispatch::{Caller, Outcome};
use starquake_kernel::handle::HandleTable;
use starquake_kernel::ipc::PortName;
use starquake_kernel::memory::testing::RecordingMapper;
use starquake_kernel::memory::{AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets};
use starquake_kernel::process::{self, Console, ProcessDescription};
use starquake_kernel::svc::{DebugSink, SvcId};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId};

use super::*;

/// The process's memory: the message buffer, the port name, the title's writable memory, its
/// read-only memory, one more page it lends at the writable memory's size, and one more range
/// it lends at the read-only memory's size.
const BASE: u64 = 0x8000_0000;
const TLS: u64 = BASE;
const NAME: u64 = BASE + PAGE_SIZE;
const AT_WRITABLE: u64 = BASE + 2 * PAGE_SIZE;
const AT_READ_ONLY: u64 = AT_WRITABLE + WRITABLE_SIZE as u64;
const AT_EXTRA: u64 = AT_READ_ONLY + READ_ONLY_SIZE as u64;
const AT_EXTRA_READ_ONLY: u64 = AT_EXTRA + WRITABLE_SIZE as u64;
const END: u64 = AT_EXTRA_READ_ONLY + READ_ONLY_SIZE as u64;

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// A Switch 2 process whose memory is all read-write, the root server with the page on the port
/// `gsvt:` as [`ROOT`], and the title's page bound through it.
struct KernelRig {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    shared: Arc<Shared>,
    /// The guest's handle to the request event's writable end.
    request: Handle,
    /// A third transfer memory the guest lent, of [`WRITABLE_SIZE`] bytes.
    extra: ObjectId,
    /// A fourth, of [`READ_ONLY_SIZE`] bytes.
    extra_read_only: ObjectId,
    /// The guest's root session.
    session: Handle,
    /// The guest's handles to its writable and read-only memories.
    memories: [Handle; 2],
}

impl KernelRig {
    fn bound() -> Self {
        let (mapper, _log) = RecordingMapper::new();
        let layout = ProcessMemory {
            code_address: BASE,
            code_size: END - BASE,
            arrangement: MemoryArrangement::FourGb,
            alias_region_extra: false,
            offsets: RegionOffsets::default(),
        };
        let mut space = AddressSpace::new(&layout, Box::new(mapper)).unwrap();
        space
            .set_process_memory_permission(BASE, END - BASE, SvcPermission::ReadWrite)
            .unwrap();
        let description = ProcessDescription::homebrew(Console::Switch2);
        let dispatcher = Dispatcher::new(process::create(&description, space).unwrap());
        let shared = fresh_shared();
        let server =
            root_server_with_page(RootState::new(), UnknownPolicy::Strict, Arc::clone(&shared))
                .with_service_id(ROOT);
        dispatcher
            .add_server(
                "gsv",
                ROOT,
                Some(PortName::new("gsvt:").unwrap()),
                Box::new(server),
            )
            .unwrap();
        let mut rig = Self {
            dispatcher,
            memory: FlatMemory::new(BASE, usize::try_from(END - BASE).unwrap()),
            env: LiveEnv::new(FixedClock::new(0, 1), 0),
            recorder: Recorder::new(),
            shared,
            request: Handle::from_raw(0),
            extra: ObjectId::from_parts(0, 0),
            extra_read_only: ObjectId::from_parts(0, 0),
            session: Handle::from_raw(0),
            memories: [Handle::from_raw(0); 2],
        };
        let writable = rig.lend(AT_WRITABLE, WRITABLE_SIZE, SvcPermission::None);
        let read_only = rig.lend(AT_READ_ONLY, READ_ONLY_SIZE, SvcPermission::Read);
        rig.memories = [writable, read_only];
        let extra = rig.lend(AT_EXTRA, WRITABLE_SIZE, SvcPermission::None);
        rig.extra = rig.object(extra);
        let extra_read_only = rig.lend(AT_EXTRA_READ_ONLY, READ_ONLY_SIZE, SvcPermission::Read);
        rig.extra_read_only = rig.object(extra_read_only);
        // 204 through the kernel, as the title sends it.
        rig.memory.write(NAME, b"gsvt:\0").unwrap();
        let session = Handle::from_register(rig.call(SvcId::ConnectToNamedPort, [0, NAME]).x[1]);
        rig.session = session;
        let mut buffer = vec![0; starquake_ipc::BUFFER_SIZE];
        let words = bind_words(writable.raw(), read_only.raw());
        let bytes = le(&words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        rig.memory.write(TLS, &buffer).unwrap();
        let regs = rig.call(SvcId::SendSyncRequest, [u64::from(session.raw()), 0]);
        assert_eq!(regs.x[0], 0);
        rig.memory.read(TLS, &mut buffer).unwrap();
        let handles = copied(&buffer);
        assert_eq!(handles.len(), 2, "204's reply");
        rig.request = Handle::from_raw(handles[0]);
        let page = rig.page();
        assert!(page.binding().is_some());
        // The handles the guest was given name the page's events.
        assert_eq!(
            rig.object(rig.request),
            page.request_event().unwrap().writable
        );
        assert_eq!(
            rig.object(Handle::from_raw(handles[1])),
            page.answer_event().unwrap()
        );
        rig
    }

    /// Makes system call `id` with `x0` and `x1`; the registers after it.
    fn call(&self, id: SvcId, x: [u64; 2]) -> GuestRegs {
        self.call4(id, [x[0], x[1], 0, 0])
    }

    /// Makes system call `id` with `x0` to `x3`; the registers after it.
    fn call4(&self, id: SvcId, x: [u64; 4]) -> GuestRegs {
        let mut regs = GuestRegs {
            tpidrro: TLS,
            ..GuestRegs::default()
        };
        regs.x[..4].copy_from_slice(&x);
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 0,
            thread: ThreadId(1),
            env: &self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
        regs
    }

    /// The guest's `CreateTransferMemory` of `size` bytes at `address`, keeping `permission`.
    fn lend(&self, address: u64, size: u32, permission: SvcPermission) -> Handle {
        let bits = u64::from(permission.bits());
        let regs = self.call4(
            SvcId::CreateTransferMemory,
            [0, address, u64::from(size), bits],
        );
        assert_eq!(regs.x[0], 0);
        Handle::from_register(regs.x[1])
    }

    /// The object a handle in the guest's table names.
    fn object(&self, handle: Handle) -> ObjectId {
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle))
            .unwrap()
    }

    /// The guest's `SignalEvent` on the request event's writable end.
    fn signal(&self) {
        let regs = self.call(SvcId::SignalEvent, [u64::from(self.request.raw()), 0]);
        assert_eq!(regs.x[0], 0);
    }

    fn page(&self) -> Page {
        self.shared.lock().page().clone()
    }

    /// The guest's `CloseHandle` of `handle`.
    fn close(&self, handle: Handle) {
        let regs = self.call(SvcId::CloseHandle, [u64::from(handle.raw()), 0]);
        assert_eq!(regs.x[0], 0);
    }

    /// The references the root keeps to the bound page's writable and read-only memories.
    fn kept(&self, binding: Binding) -> [u32; 2] {
        [
            self.dispatcher.kept_transfer_memory(ROOT, binding.writable),
            self.dispatcher.kept_transfer_memory(ROOT, binding.read_only),
        ]
    }

    /// Plays the timer thread's turn at `now` over the kernel's dispatcher: the drain, then the
    /// page turn.
    fn turn(&self, now: HostNanos) {
        let mut host = OnTheKernel {
            dispatcher: &self.dispatcher,
            memory: &self.memory,
        };
        self.shared.drain(&mut host);
        self.shared.tick_page(&mut host, now);
    }

    /// The `u64` at `offset` of the title's memory starting at `base`.
    fn word(&self, base: u64, offset: u64) -> u64 {
        let mut bytes = [0; 8];
        self.memory.read(base + offset, &mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }

    /// The result word of the guest's `CreateTransferMemory` of its writable memory's range
    /// again: refused while a lend still locks it, taken once none does.
    fn lend_writable_again(&self) -> u64 {
        let bits = u64::from(SvcPermission::None.bits());
        self.call4(
            SvcId::CreateTransferMemory,
            [0, AT_WRITABLE, u64::from(WRITABLE_SIZE), bits],
        )
        .x[0]
    }
}

/// The timer thread's host over the kernel's dispatcher, as the wiring's is
/// (`starquake-core`'s `hle::graphics`): the page's memories are read and written through the
/// dispatcher, and the root's references dropped and its sessions counted for [`ROOT`].
struct OnTheKernel<'a> {
    dispatcher: &'a Dispatcher,
    memory: &'a FlatMemory,
}

impl TimerHost for OnTheKernel<'_> {
    fn record(&mut self, _: EventKind) {}

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.dispatcher.unlock_device_share(token)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        let env = LiveEnv::new(FixedClock::new(0, 1), 0);
        self.dispatcher
            .signal_event(event, &env, &Recorder::new())
            .map_err(|_| KernelError::InvalidState)
    }

    fn read_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        out: &mut [u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .read_transfer_memory(self.memory, memory, offset, out)
    }

    fn write_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .write_transfer_memory(self.memory, memory, offset, data)
    }

    fn release_kept_transfer_memory(&mut self, memory: ObjectId) -> SvcResult<()> {
        self.dispatcher.release_kept_transfer_memory(ROOT, memory)
    }

    fn root_sessions(&mut self) -> SvcResult<usize> {
        Ok(self.dispatcher.open_sessions(ROOT))
    }
}

#[test]
fn the_page_answers_after_the_title_closes_its_handles_and_the_root_s_close_lets_it_go() {
    let rig = KernelRig::bound();
    let binding = rig.page().binding().unwrap();
    assert_eq!(rig.kept(binding), [1, 1]);
    // The title closes both its handles: the root's references keep the memories, still
    // locked.
    rig.close(rig.memories[0]);
    rig.close(rig.memories[1]);
    assert_eq!(rig.kept(binding), [1, 1]);
    assert!(rig.dispatcher.transfer_memory(binding.writable).is_ok());
    assert!(rig.dispatcher.transfer_memory(binding.read_only).is_ok());
    assert_eq!(
        rig.lend_writable_again(),
        u64::from(KernelError::InvalidCurrentMemory.code().raw())
    );
    // It asks about a syncpoint whose counter moved, and signals: the answer is written into
    // its memories through the kernel.
    let value = rig.shared.syncpoints().increment(id(SP));
    let group = usize::from(SP) / GROUP_BITS;
    let bit = 1u64 << (usize::from(SP) % GROUP_BITS);
    rig.memory
        .write(
            AT_WRITABLE + group_offset(SLOT_REQUEST_OFFSET, group),
            &bit.to_le_bytes(),
        )
        .unwrap();
    rig.memory
        .write(
            AT_WRITABLE + u64::from(GROUP_REQUEST_OFFSET),
            &(1u64 << group).to_le_bytes(),
        )
        .unwrap();
    rig.signal();
    rig.turn(T0);
    assert_eq!(
        rig.word(AT_READ_ONLY, value_offset(id(SP))),
        u64::from(value)
    );
    assert_eq!(
        rig.word(AT_READ_ONLY, group_offset(SLOT_ACK_OFFSET, group)),
        bit
    );
    assert_eq!(
        rig.word(AT_READ_ONLY, u64::from(GROUP_ACK_OFFSET)),
        1 << group
    );
    // The title closes its root session, the only one: the drain drops the root's references,
    // the memories go, and the title's pages are its own again.
    rig.close(rig.session);
    assert_eq!(rig.dispatcher.open_sessions(ROOT), 0);
    rig.turn(T0);
    assert_eq!(rig.kept(binding), [0, 0]);
    assert_eq!(
        rig.dispatcher.transfer_memory(binding.writable),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(
        rig.dispatcher.transfer_memory(binding.read_only),
        Err(KernelError::InvalidHandle)
    );
    assert_eq!(rig.page().binding(), None);
    assert_eq!(rig.lend_writable_again(), 0);
}

#[test]
fn the_root_s_close_leaves_the_title_s_own_hold_on_its_memories() {
    let rig = KernelRig::bound();
    let binding = rig.page().binding().unwrap();
    rig.close(rig.session);
    rig.turn(T0);
    assert_eq!(rig.kept(binding), [0, 0]);
    assert_eq!(rig.page().binding(), None);
    // The title still holds its handles: the memories stay, locked, until it closes them.
    assert!(rig.dispatcher.transfer_memory(binding.writable).is_ok());
    assert_eq!(
        rig.lend_writable_again(),
        u64::from(KernelError::InvalidCurrentMemory.code().raw())
    );
    rig.close(rig.memories[0]);
    assert_eq!(rig.lend_writable_again(), 0);
}

#[test]
fn a_page_restored_on_the_kernel_hears_the_guest_s_signal() {
    let rig = KernelRig::bound();
    let saved = rig.shared.save_page().unwrap();
    let request = rig.page().request_event().unwrap().writable;
    // What a restored kernel holds: the events and memories, and no hook. The guest's signal
    // reaches no queue.
    rig.dispatcher.forget_event_hooks();
    rig.signal();
    assert_eq!(rig.shared.queue().take_all(), []);
    // The restore re-attaches the hook, to the restored state's queue.
    let restored = fresh_shared();
    restored
        .restore_page(&saved, &rig.dispatcher, ROOT)
        .unwrap();
    assert_eq!(restored.lock().page(), &rig.page());
    rig.signal();
    assert_eq!(
        restored.queue().take_all(),
        [Completion::GuestSignalled { event: request }]
    );
    assert_eq!(rig.shared.queue().take_all(), []);
}

/// The bound rig's page corrupted to fail each of the restore's kernel checks, with the error.
fn corrupt_pages(rig: &KernelRig) -> Vec<(Page, PageSnapshotError)> {
    let page = rig.page();
    let request = page.request_event().unwrap();
    let binding = page.binding().unwrap();
    // The guest's own event: its writable end.
    let guests = rig.object(Handle::from_register(
        rig.call(SvcId::CreateEvent, [0, 0]).x[1],
    ));
    let never_made = object(0x7FFF, 9);
    let corrupt = |change: &dyn Fn(&mut Page)| {
        let mut page = page.clone();
        change(&mut page);
        page
    };
    let cases = [
        // The request event's two ends swapped: the hook is asked for the readable end.
        (
            corrupt(&|page| {
                page.request = Some(RequestEvent {
                    writable: request.readable,
                    readable: request.writable,
                });
            }),
            PageSnapshotError::Rehook(KernelError::InvalidHandle),
        ),
        (
            corrupt(&|page| page.answer = Some(guests)),
            PageSnapshotError::AnswerEvent,
        ),
        (
            corrupt(&|page| page.answer = Some(never_made)),
            PageSnapshotError::AnswerEvent,
        ),
        // A live object the root did not make: one of the title's memories.
        (
            corrupt(&|page| page.answer = Some(rig.extra)),
            PageSnapshotError::AnswerEvent,
        ),
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    writable: guests,
                    ..binding
                });
            }),
            PageSnapshotError::WritableMemory(KernelError::InvalidHandle),
        ),
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    writable: binding.read_only,
                    read_only: binding.writable,
                });
            }),
            PageSnapshotError::WritableSize {
                size: u64::from(READ_ONLY_SIZE),
            },
        ),
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    read_only: never_made,
                    ..binding
                });
            }),
            PageSnapshotError::ReadOnlyMemory(KernelError::InvalidHandle),
        ),
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    read_only: rig.extra,
                    ..binding
                });
            }),
            PageSnapshotError::ReadOnlySize {
                size: u64::from(WRITABLE_SIZE),
            },
        ),
        // Live memories of the right sizes that the root does not keep.
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    writable: rig.extra,
                    ..binding
                });
            }),
            PageSnapshotError::WritableNotKept,
        ),
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    read_only: rig.extra_read_only,
                    ..binding
                });
            }),
            PageSnapshotError::ReadOnlyNotKept,
        ),
        // Both unkept: the writable memory's is reported.
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    writable: rig.extra,
                    read_only: rig.extra_read_only,
                });
            }),
            PageSnapshotError::WritableNotKept,
        ),
        // Unkept and the wrong size: the size is reported.
        (
            corrupt(&|page| {
                page.binding = Some(Binding {
                    writable: rig.extra_read_only,
                    ..binding
                });
            }),
            PageSnapshotError::WritableSize {
                size: u64::from(READ_ONLY_SIZE),
            },
        ),
    ];
    cases.into()
}

#[test]
fn each_kernel_check_refuses_on_the_kernel_as_on_the_fake() {
    let rig = KernelRig::bound();
    let page = rig.page();
    let cases = corrupt_pages(&rig);
    // The double holds what the kernel holds: the page's objects, and the extra memory.
    let mut fake = Holding::of(&page);
    fake.memories.insert(rig.extra, u64::from(WRITABLE_SIZE));
    fake.memories
        .insert(rig.extra_read_only, u64::from(READ_ONLY_SIZE));
    rig.dispatcher.forget_event_hooks();
    for (corrupted, error) in cases {
        let bytes = snapshot_of(&corrupted);
        let real = fresh_shared();
        assert_eq!(
            real.restore_page(&bytes, &rig.dispatcher, ROOT),
            Err(error.clone()),
            "{error}"
        );
        assert_eq!(real.lock().page(), &Page::default(), "{error}");
        assert_eq!(
            fresh_shared().restore_page(&bytes, &fake, ROOT),
            Err(error.clone()),
            "{error}"
        );
        // Nothing was attached: the guest's signal reaches no queue.
        rig.signal();
        assert_eq!(real.queue().take_all(), [], "{error}");
    }
    assert!(fake.attached_nothing());
    // The page as saved is taken by both.
    let bytes = snapshot_of(&page);
    fresh_shared()
        .restore_page(&bytes, &rig.dispatcher, ROOT)
        .unwrap();
    fresh_shared().restore_page(&bytes, &fake, ROOT).unwrap();
}

#[test]
fn the_fake_kernel_refuses_a_rehook_as_the_real_one() {
    // The kernel's dispatcher has no server for the root and so made no event; the double holds
    // the events for another service. A page with its events and no binding: both refuse the
    // answer event before the hook.
    let (dispatcher, _) = crate::completion::testing::real_kernel();
    let page = Page {
        request: Some(RequestEvent {
            writable: object(0x11, 1),
            readable: object(0x12, 2),
        }),
        answer: Some(object(0x13, 3)),
        ..Page::default()
    };
    let other = ServiceId(ROOT.0 + 1);
    let fake = Holding {
        hooked: [(other, object(0x11, 1))].into(),
        events: [(other, object(0x12, 2)), (other, object(0x13, 3))].into(),
        ..Holding::default()
    };
    let bytes = snapshot_of(&page);
    let real = fresh_shared().restore_page(&bytes, &dispatcher, ROOT);
    assert_eq!(real, fresh_shared().restore_page(&bytes, &fake, ROOT));
    assert_eq!(real, Err(PageSnapshotError::AnswerEvent));
    // Without the answer event, the hook is what is refused, by both.
    let page = Page {
        answer: None,
        ..page
    };
    let bytes = snapshot_of(&page);
    let real = fresh_shared().restore_page(&bytes, &dispatcher, ROOT);
    assert_eq!(real, fresh_shared().restore_page(&bytes, &fake, ROOT));
    assert_eq!(
        real,
        Err(PageSnapshotError::Rehook(KernelError::InvalidHandle))
    );
}
