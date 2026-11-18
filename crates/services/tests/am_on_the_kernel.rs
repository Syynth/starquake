//! `appletOE` on the real kernel: libnx's `SetFocusHandlingMode(0, 0, 1)` clears the message
//! event the guest holds a handle to, `ISelfController::Exit` answers the guest and then
//! ends its process, releasing everything the process held, and `IApplicationFunctions` 210
//! copies a handle the guest can wait on, to the same event on every call. And restoring
//! beside the kernel: the server, saved, restores beside the kernel it ran on, and the
//! kernel answers a restore's questions as the services framework's fake kernel does.
//!
//! The caller is a thread the scheduler runs (`run_a_user_thread`), so the kernel writes a reply
//! only while that thread has not ended: a kernel that ended the process before writing the
//! reply would leave the request in the message buffer, and
//! `exit_replies_and_then_the_kernel_ends_the_process` would fail.

/// Everything in a `cfg(test)` module, so that the helpers may unwrap as the tests do.
#[cfg(test)]
mod on_the_kernel {
    use std::sync::Arc;

    use starquake_cpu::GuestRegs;
    use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
    use starquake_kernel::dispatch::{Caller, Dispatcher, Outcome, Tables};
    use starquake_kernel::handle::{Handle, HandleTable};
    use starquake_kernel::id::{ObjectId, ProcessId};
    use starquake_kernel::ipc::{EventHook, PortName};
    use starquake_kernel::memory::testing::RecordingMapper;
    use starquake_kernel::memory::{
        AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets, SharedMemory, SvcPermission,
    };
    use starquake_kernel::object::Object;
    use starquake_kernel::process::{
        self, Console, ProcessDescription, ProcessState, VIRTUAL_TO_PHYSICAL_CORE,
    };
    use starquake_kernel::result::{KernelError, ResultCode};
    use starquake_kernel::sched::{CoreId, CoreSignals, Priority, UserThread};
    use starquake_kernel::svc::{DebugSink, SvcId};
    use starquake_kernel::tables::Scheduler;
    use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
    use starquake_services::am::command::application_functions::GET_UNKNOWN_EVENT_210;
    use starquake_services::am::command::common_state_getter::GET_EVENT_HANDLE;
    use starquake_services::am::command::proxy::{
        GET_APPLICATION_FUNCTIONS, GET_COMMON_STATE_GETTER, GET_SELF_CONTROLLER,
    };
    use starquake_services::am::command::proxy_service::OPEN_APPLICATION_PROXY;
    use starquake_services::am::command::self_controller::{EXIT, SET_FOCUS_HANDLING_MODE};
    use starquake_services::am::{Applet, AppletSettings, OE_NAME, oe_server};
    use starquake_services::framework::fake::FakeKernel;
    use starquake_services::framework::{
        Kernel, RestoredKernel, Server, SnapshotError, UnknownPolicy,
    };
    use starquake_trace::{
        EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId, VecSink,
    };

    /// The process's memory: two pages the guest may read and write, the message buffer and
    /// then the port name.
    const BASE: u64 = 0x8000_0000;
    const PAGES: u64 = 2;
    const TLS: u64 = BASE;
    const NAME: u64 = BASE + PAGE_SIZE;
    /// Where the guest keeps the handle array it waits on: after the port name.
    const WAITED: u64 = NAME + 0x100;
    /// The number the test registers `appletOE`'s server under.
    const OE: ServiceId = ServiceId(0x64);
    /// The calling guest thread, and the core it runs on.
    const THREAD: ThreadId = ThreadId(1);
    const CORE: u8 = 0;

    struct NoDebug;

    impl DebugSink for NoDebug {
        fn output(&mut self, _: GuestTicks, _: &[u8]) {}
    }

    /// A Switch 2 process with its pages read-write, `appletOE` on a port of its name, the
    /// guest's memory and the events recorded.
    struct Rig {
        dispatcher: Dispatcher,
        process: ProcessId,
        memory: FlatMemory,
        env: LiveEnv,
        recorder: Recorder,
        events: Arc<VecSink>,
    }

    /// A reply: its result, copied and moved handles, and data.
    struct Answer {
        result: u32,
        copies: Vec<u32>,
        moves: Vec<u32>,
    }

    impl Rig {
        fn new() -> Self {
            let (mapper, _log) = RecordingMapper::new();
            let memory = ProcessMemory {
                code_address: BASE,
                code_size: PAGES * PAGE_SIZE,
                arrangement: MemoryArrangement::FourGb,
                alias_region_extra: false,
                offsets: RegionOffsets::default(),
            };
            let mut space = AddressSpace::new(&memory, Box::new(mapper)).unwrap();
            space
                .set_process_memory_permission(BASE, PAGES * PAGE_SIZE, SvcPermission::ReadWrite)
                .unwrap();
            let description = ProcessDescription::homebrew(Console::Switch2);
            let mut tables = process::create(&description, space).unwrap();
            let process = tables.processes.own();
            run_a_user_thread(&mut tables, process, &description);
            let dispatcher = Dispatcher::new(tables);
            let server = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict);
            dispatcher
                .add_server(
                    OE_NAME,
                    OE,
                    Some(PortName::new(OE_NAME).unwrap()),
                    Box::new(server),
                )
                .unwrap();
            let recorder = Recorder::new();
            let events = Arc::new(VecSink::new());
            recorder.add_sink(Arc::clone(&events));
            Self {
                dispatcher,
                process,
                memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
                env: LiveEnv::new(FixedClock::new(0, 1), 0),
                recorder,
                events,
            }
        }

        /// Makes system call `id` with `x0` to `x3`; how the dispatch ended, and the
        /// registers after it.
        fn dispatch(&self, id: SvcId, x: [u64; 4]) -> (Outcome, GuestRegs) {
            let mut regs = GuestRegs {
                tpidrro: TLS,
                ..GuestRegs::default()
            };
            regs.x[..4].copy_from_slice(&x);
            let mut debug = NoDebug;
            let mut caller = Caller {
                core: CORE,
                thread: THREAD,
                env: &self.env,
                memory: &self.memory,
                debug: &mut debug,
                recorder: &self.recorder,
            };
            let record = self
                .dispatcher
                .dispatch(id.number(), &mut regs, &mut caller);
            (record.outcome, regs)
        }

        /// [`Rig::dispatch`] for a call that completes: the registers after it.
        fn call(&self, id: SvcId, x: [u64; 4]) -> GuestRegs {
            let (outcome, regs) = self.dispatch(id, x);
            assert_eq!(outcome, Outcome::Completed, "{}", id.name());
            regs
        }

        fn connect(&self) -> Handle {
            let mut name = OE_NAME.as_bytes().to_vec();
            name.push(0);
            self.memory.write(NAME, &name).unwrap();
            let regs = self.call(SvcId::ConnectToNamedPort, [0, NAME, 0, 0]);
            assert_eq!(regs.x[0], 0);
            Handle::from_register(regs.x[1])
        }

        /// Puts `request` in the message buffer and sends it on `session`; how the dispatch
        /// ended, the registers, and the reply in the message buffer.
        fn exchange(&self, session: Handle, request: &RequestBuilder) -> (Outcome, u64, Answer) {
            let mut buffer = [0; BUFFER_SIZE];
            request.build(&mut buffer).unwrap();
            self.memory.write(TLS, &buffer).unwrap();
            let (outcome, regs) =
                self.dispatch(SvcId::SendSyncRequest, [u64::from(session.raw()), 0, 0, 0]);
            self.memory.read(TLS, &mut buffer).unwrap();
            let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
            let Body::CmifReply(reply) = &message.body else {
                unreachable!("not a CMIF reply");
            };
            let answer = Answer {
                result: reply.header.result,
                copies: message.copy_handles.clone(),
                moves: message.move_handles.clone(),
            };
            (outcome, regs.x[0], answer)
        }

        /// [`Rig::exchange`] for a request that is answered with success at once.
        fn send(&self, session: Handle, request: &RequestBuilder) -> Answer {
            let (outcome, x0, answer) = self.exchange(session, request);
            assert_eq!((outcome, x0), (Outcome::Completed, 0));
            assert_eq!(answer.result, ResultCode::SUCCESS.raw());
            answer
        }

        /// Opens the proxy as libnx does, then the interface `command` returns from it.
        fn open(&self, command: u32) -> Handle {
            let root = self.connect();
            let open = RequestBuilder::cmif(CommandType::Request, OPEN_APPLICATION_PROXY)
                .send_pid(0)
                .copy_handle(Handle::CURRENT_PROCESS.raw())
                .data(&0_u64.to_le_bytes());
            let proxy = Handle::from_raw(self.send(root, &open).moves[0]);
            let opened = self.send(proxy, &RequestBuilder::cmif(CommandType::Request, command));
            Handle::from_raw(opened.moves[0])
        }

        /// The message event's readable end, as `GetEventHandle` copies it to the guest.
        fn message_event(&self) -> Handle {
            let getter = self.open(GET_COMMON_STATE_GETTER);
            let request = RequestBuilder::cmif(CommandType::Request, GET_EVENT_HANDLE);
            Handle::from_raw(self.send(getter, &request).copies[0])
        }

        /// `ResetSignal` on `handle`: success when the event was signalled, Horizon's
        /// "invalid state" when it was not.
        fn reset(&self, handle: Handle) -> u64 {
            self.call(SvcId::ResetSignal, [u64::from(handle.raw()), 0, 0, 0])
                .x[0]
        }

        /// `WaitSynchronization` on `handle` alone, with a timeout of zero (a poll): its
        /// result (`x0`).
        fn poll(&self, handle: Handle) -> u64 {
            self.memory
                .write(WAITED, &handle.raw().to_le_bytes())
                .unwrap();
            self.call(SvcId::WaitSynchronization, [0, WAITED, 1, 0]).x[0]
        }

        /// The object `handle` names in the process's handle table.
        fn object_of(&self, handle: Handle) -> starquake_kernel::id::ObjectId {
            self.dispatcher
                .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle).unwrap())
        }

        /// `SetFocusHandlingMode(notify, background, suspend)` on `controller`.
        fn set_focus_handling_mode(&self, controller: Handle, flags: [u8; 3]) {
            let request =
                RequestBuilder::cmif(CommandType::Request, SET_FOCUS_HANDLING_MODE).data(&flags);
            self.send(controller, &request);
        }

        /// What the process holds: its handles, every object in the table, and whether it
        /// has ended.
        fn holdings(&self) -> (usize, usize, bool) {
            let process = self.process;
            self.dispatcher
                .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
                    (
                        handles.len(),
                        scheduler.objects().len(),
                        scheduler.process(process).unwrap().state() == ProcessState::Terminated,
                    )
                })
        }
    }

    /// Signals nobody listens to: the rig switches its one core by hand.
    struct Quiet;

    impl CoreSignals for Quiet {
        fn interrupt(&mut self, _: CoreId) {}

        fn wake(&mut self, _: CoreId) {}
    }

    /// Makes [`CORE`] run a main thread of `process` as `description` describes it, as the core
    /// loop runs it, so
    /// the caller is a thread the scheduler knows: one whose reply the kernel writes only while
    /// it has not ended. Without it the kernel only reports the caller, and writes its reply
    /// whether or not the process has ended, so the order of reply and end would not show.
    /// The thread's object keeps its creator's reference, which the test never lets go.
    fn run_a_user_thread(
        tables: &mut Tables,
        process: ProcessId,
        description: &ProcessDescription,
    ) {
        // The main thread's virtual core is the physical core the rig calls from.
        let virtual_core = description.main_thread_core;
        let physical = VIRTUAL_TO_PHYSICAL_CORE[usize::from(virtual_core)];
        assert_eq!(physical, CORE);
        let core = CoreId::new(CORE).unwrap();
        let spec = UserThread {
            entry: BASE,
            argument: 0,
            stack_top: BASE + PAGES * PAGE_SIZE,
            priority: Priority::new(description.main_thread_priority).unwrap(),
            virtual_core,
        };
        let scheduler = &mut tables.scheduler;
        let (id, _) = scheduler
            .create_user_thread(&mut tables.address_space, process, &spec)
            .unwrap();
        scheduler.start(id).unwrap();
        scheduler.update(GuestTicks(0), None, &mut Quiet).unwrap();
        assert_eq!(scheduler.schedule(core, &mut Quiet), Ok(Some(id)));
    }

    fn invalid_state() -> u64 {
        u64::from(KernelError::InvalidState.code().raw())
    }

    #[test]
    fn the_message_event_is_signalled_from_launch() {
        let rig = Rig::new();
        let event = rig.message_event();
        assert_eq!(rig.reset(event), 0);
        assert_eq!(rig.reset(event), invalid_state());
    }

    #[test]
    fn set_focus_handling_mode_0_0_1_clears_the_message_event_the_guest_holds() {
        let rig = Rig::new();
        let event = rig.message_event();
        let controller = rig.open(GET_SELF_CONTROLLER);
        rig.set_focus_handling_mode(controller, [0, 0, 1]);
        assert_eq!(rig.reset(event), invalid_state());
        // Notifications on again: the pending focus change signals it again.
        rig.set_focus_handling_mode(controller, [1, 0, 1]);
        assert_eq!(rig.reset(event), 0);
    }

    /// The handle 210 copies names a synchronisation object the guest can wait on (a poll of it
    /// times out, where a handle that is not one is `InvalidHandle`), an event never signalled
    /// (`ResetSignal` finds it clear), and two calls copy two handles to the same event.
    #[test]
    fn get_unknown_event_210_copies_a_handle_the_guest_waits_on() {
        let rig = Rig::new();
        let functions = rig.open(GET_APPLICATION_FUNCTIONS);
        let request = RequestBuilder::cmif(CommandType::Request, GET_UNKNOWN_EVENT_210);
        let first = Handle::from_raw(rig.send(functions, &request).copies[0]);
        let second = Handle::from_raw(rig.send(functions, &request).copies[0]);
        assert_ne!(first, second);
        assert_eq!(rig.object_of(first), rig.object_of(second));
        let timed_out = u64::from(KernelError::TimedOut.code().raw());
        for handle in [first, second] {
            assert_eq!(rig.poll(handle), timed_out);
            assert_eq!(rig.reset(handle), invalid_state());
        }
        // A client session is no synchronisation object: its poll is refused, not timed out.
        let invalid = u64::from(KernelError::InvalidHandle.code().raw());
        assert_eq!(rig.poll(functions), invalid);
    }

    #[test]
    fn exit_replies_and_then_the_kernel_ends_the_process() {
        let rig = Rig::new();
        rig.message_event();
        let controller = rig.open(GET_SELF_CONTROLLER);
        // Six sessions (the port's twice, the two proxies, the getter and the controller), each
        // held by one handle, and the copied event; besides them the table holds the process,
        // its resource limit, the event the service keeps the creator's reference to, and the
        // calling thread, held by its creator's reference (`run_a_user_thread`).
        let sessions = 6;
        assert_eq!(rig.holdings(), (sessions + 1, sessions + 4, false));
        let before = rig.events.events().len();

        let (outcome, x0, answer) = rig.exchange(
            controller,
            &RequestBuilder::cmif(CommandType::Request, EXIT),
        );
        assert_eq!(outcome, Outcome::ProcessExited);
        // The reply reached the guest's message buffer and its result its `x0`, and the
        // request was recorded: the service's event, then the system call's.
        assert_eq!(answer.result, ResultCode::SUCCESS.raw());
        assert_eq!(x0, 0);
        let recorded: Vec<EventKind> = rig.events.events()[before..]
            .iter()
            .map(|event| event.kind)
            .collect();
        assert!(
            matches!(
                recorded.as_slice(),
                [
                    EventKind::Ipc { command, result, .. },
                    EventKind::Svc { id, result: svc, .. },
                ] if *command == EXIT
                    && result.0 == 0
                    && *id == u32::from(SvcId::SendSyncRequest.number())
                    && svc.0 == 0
            ),
            "{recorded:?}"
        );
        // Then the process ended: no handle is left, and of the objects only the process, its
        // resource limit, the service's event and the ended thread's creator's reference
        // remain.
        assert_eq!(rig.holdings(), (0, 4, true));
    }

    // ----- restoring beside the kernel ---------------------------------------------------------

    /// What `appletOE`'s server saves now.
    fn saved_oe(dispatcher: &Dispatcher) -> Vec<u8> {
        let mut saved = None;
        let out = dispatcher.inspect_handlers(|service, handler| {
            if service == OE {
                let handler: &dyn core::any::Any = handler;
                saved = Some(handler.downcast_ref::<Server>().unwrap().save().unwrap());
            }
        });
        assert_eq!(out, 0);
        saved.unwrap()
    }

    /// `appletOE`'s server as a run builds it: known by the number the kernel has it under.
    fn oe_as_wired() -> Server {
        oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict).with_service_id(OE)
    }

    /// The production arrangement: `appletOE` run on the kernel, saved, and restored beside the
    /// same kernel into a server built as a run builds it, keeps the event the guest holds a
    /// handle to; a server known by another number, and `restore` with no kernel, refuse the
    /// same snapshot with the exact errors.
    #[test]
    fn appletoe_restores_beside_the_kernel_it_ran_on() {
        let rig = Rig::new();
        let event = rig.object_of(rig.message_event());
        let saved = saved_oe(&rig.dispatcher);

        let mut restored = oe_as_wired();
        assert_eq!(restored.restore_beside(&saved, &rig.dispatcher), Ok(()));
        assert_eq!(
            restored.family::<Applet>().unwrap().message_event(),
            Some(event)
        );

        let mut elsewhere = oe_server(AppletSettings::eden_default(), UnknownPolicy::Strict)
            .with_service_id(ServiceId(OE.0 + 1));
        assert_eq!(
            elsewhere.restore_beside(&saved, &rig.dispatcher),
            Err(SnapshotError::FamilyObject(
                "its applet's message event is not a live readable event its server made"
            ))
        );
        assert_eq!(
            oe_as_wired().restore(&saved),
            Err(SnapshotError::NeedsKernel)
        );
    }

    /// Another server's number, for the parity test's second server.
    const OTHER: ServiceId = ServiceId(0x65);

    /// What a restore asks of a kernel about each object, by name: whether it is a live
    /// readable event of [`OE`]'s and of [`OTHER`]'s, and the size and permission of the shared
    /// memory it names, if any.
    type Answers = Vec<(&'static str, bool, bool, Option<(u64, SvcPermission)>)>;

    /// The questions of [`Answers`] about `objects`, asked of `kernel`.
    fn answers(kernel: &dyn RestoredKernel, objects: &[(&'static str, ObjectId)]) -> Answers {
        objects
            .iter()
            .map(|&(name, id)| {
                let memory = kernel
                    .shared_memory(id)
                    .map(|memory| (memory.size(), memory.user_permission()));
                (
                    name,
                    kernel.is_service_readable_event(OE, id),
                    kernel.is_service_readable_event(OTHER, id),
                    memory,
                )
            })
            .collect()
    }

    /// A hook that does nothing.
    struct QuietHook;

    impl EventHook for QuietHook {
        fn signalled(&self, _: ObjectId) {}
    }

    /// The fake kernel answers a restore as the kernel does (the fake's module docs, "A
    /// restore's questions"): the same objects made on each (a readable event of each of two
    /// servers, both ends of an event one of them made, a read-only and a read-write shared
    /// memory, and an id that names nothing live) get the same answers, which are these.
    #[test]
    fn the_fake_and_the_kernel_answer_a_restore_alike() {
        let rig = Rig::new();
        rig.dispatcher
            .add_server(
                "other",
                OTHER,
                None,
                Box::new(Server::new("other", UnknownPolicy::Strict)),
            )
            .unwrap();
        let mine = rig.object_of(rig.message_event());
        let made = rig
            .dispatcher
            .with::<(&mut AddressSpace, &mut Scheduler), _>(|(space, scheduler)| {
                let ipc = scheduler.ipc();
                let (oe, other) = (ipc.service(OE).unwrap(), ipc.service(OTHER).unwrap());
                let others = scheduler.create_readable_event().unwrap();
                let (writable, readable) = scheduler.create_event().unwrap();
                let gone = scheduler.create_readable_event().unwrap();
                let (_, ipc) = scheduler.ipc_mut();
                ipc.add_server_event(other, others).unwrap();
                for event in [writable, readable, gone] {
                    ipc.add_server_event(oe, event).unwrap();
                }
                scheduler.objects_mut().release(gone).unwrap();
                let mut memory = |permission| {
                    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
                    let memory = SharedMemory::new(pages.backing(), permission).unwrap();
                    scheduler
                        .objects_mut()
                        .create(Object::SharedMemory(memory))
                        .unwrap()
                };
                let read = memory(SvcPermission::Read);
                let write = memory(SvcPermission::ReadWrite);
                [
                    ("its own event", mine),
                    ("another server's event", others),
                    ("a writable end it made", writable),
                    ("the readable end of that event", readable),
                    ("a read-only shared memory", read),
                    ("a read-write shared memory", write),
                    ("an event that is gone", gone),
                ]
            });

        let mut fake = FakeKernel::new(0).unwrap();
        fake.set_current_service(OE);
        let mine = fake.create_event().unwrap();
        fake.set_current_service(OTHER);
        let others = fake.create_event().unwrap();
        fake.set_current_service(OE);
        let (writable, readable) = fake.create_hooked_event(Arc::new(QuietHook)).unwrap();
        let read = fake.create_shared_memory(PAGE_SIZE, SvcPermission::Read);
        let write = fake.create_shared_memory(PAGE_SIZE, SvcPermission::ReadWrite);
        let faked = [
            ("its own event", mine),
            ("another server's event", others),
            ("a writable end it made", writable),
            ("the readable end of that event", readable),
            ("a read-only shared memory", read.unwrap().0),
            ("a read-write shared memory", write.unwrap().0),
            ("an event that is gone", fake.new_object()),
        ];

        let expected: Answers = vec![
            ("its own event", true, false, None),
            ("another server's event", false, true, None),
            ("a writable end it made", false, false, None),
            ("the readable end of that event", true, false, None),
            (
                "a read-only shared memory",
                false,
                false,
                Some((PAGE_SIZE, SvcPermission::Read)),
            ),
            (
                "a read-write shared memory",
                false,
                false,
                Some((PAGE_SIZE, SvcPermission::ReadWrite)),
            ),
            ("an event that is gone", false, false, None),
        ];
        assert_eq!(answers(&rig.dispatcher, &made), expected);
        assert_eq!(answers(&fake, &faked), expected);
    }
}
