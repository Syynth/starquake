//! `appletOE` on the real kernel: libnx's `SetFocusHandlingMode(0, 0, 1)` clears the message
//! event the guest holds a handle to, and `ISelfController::Exit` answers the guest and then
//! ends its process, releasing everything the process held.

/// Everything in a `cfg(test)` module, so that the helpers may unwrap as the tests do.
#[cfg(test)]
mod on_the_kernel {
    use std::sync::Arc;

    use starquake_cpu::GuestRegs;
    use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
    use starquake_kernel::dispatch::{Caller, Dispatcher, Outcome};
    use starquake_kernel::handle::{Handle, HandleTable};
    use starquake_kernel::id::ProcessId;
    use starquake_kernel::ipc::PortName;
    use starquake_kernel::memory::testing::RecordingMapper;
    use starquake_kernel::memory::{
        AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets, SvcPermission,
    };
    use starquake_kernel::process::{self, Console, ProcessDescription, ProcessState};
    use starquake_kernel::result::{KernelError, ResultCode};
    use starquake_kernel::svc::{DebugSink, SvcId};
    use starquake_kernel::tables::Scheduler;
    use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
    use starquake_services::am::command::common_state_getter::GET_EVENT_HANDLE;
    use starquake_services::am::command::proxy::{GET_COMMON_STATE_GETTER, GET_SELF_CONTROLLER};
    use starquake_services::am::command::proxy_service::OPEN_APPLICATION_PROXY;
    use starquake_services::am::command::self_controller::{EXIT, SET_FOCUS_HANDLING_MODE};
    use starquake_services::am::{AppletSettings, OE_NAME, oe_server};
    use starquake_services::framework::UnknownPolicy;
    use starquake_trace::{
        EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId, VecSink,
    };

    /// The process's memory: two pages the guest may read and write, the message buffer and
    /// then the port name.
    const BASE: u64 = 0x8000_0000;
    const PAGES: u64 = 2;
    const TLS: u64 = BASE;
    const NAME: u64 = BASE + PAGE_SIZE;
    /// The number the test registers `appletOE`'s server under.
    const OE: ServiceId = ServiceId(0x64);
    /// The calling guest thread.
    const THREAD: ThreadId = ThreadId(1);

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
            let tables = process::create(&description, space).unwrap();
            let process = tables.processes.own();
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
                core: 0,
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

    #[test]
    fn exit_replies_and_then_the_kernel_ends_the_process() {
        let rig = Rig::new();
        rig.message_event();
        let controller = rig.open(GET_SELF_CONTROLLER);
        // Six sessions (the port's twice, the two proxies, the getter and the controller), each
        // held by one handle, and the copied event; besides them the table holds the process
        // and the event the service keeps the creator's reference to.
        let sessions = 6;
        assert_eq!(rig.holdings(), (sessions + 1, sessions + 2, false));
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
        // Then the process ended: no handle is left, and of the objects only the process and
        // the service's event remain.
        assert_eq!(rig.holdings(), (0, 2, true));
    }
}
