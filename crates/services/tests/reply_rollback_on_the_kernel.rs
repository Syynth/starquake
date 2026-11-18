//! The framework's reply builder on the real kernel, when the caller's handle table fills up
//! part-way through a reply: the builder answers with the kernel's error and names no handle,
//! and the kernel takes back the handles and the session already placed (kernel
//! `dispatch/ipc/placed.rs`, issue #209). A reply that fits keeps every handle it names.

/// Everything in a `cfg(test)` module, so that the helpers may unwrap as the tests do.
#[cfg(test)]
mod on_the_kernel {
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    use starquake_cpu::GuestRegs;
    use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
    use starquake_kernel::dispatch::{Caller, Dispatcher, Outcome};
    use starquake_kernel::handle::{Handle, HandleTable};
    use starquake_kernel::id::ObjectId;
    use starquake_kernel::ipc::PortName;
    use starquake_kernel::memory::testing::RecordingMapper;
    use starquake_kernel::memory::{
        AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets, SvcPermission,
    };
    use starquake_kernel::process::{self, Console, ProcessDescription};
    use starquake_kernel::result::{KernelError, ResultCode};
    use starquake_kernel::svc::{DebugSink, SvcId};
    use starquake_kernel::tables::Scheduler;
    use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
    use starquake_services::commands;
    use starquake_services::framework::{
        Context, Outcome as MethodOutcome, Reply, Request, Server, UnknownPolicy,
    };
    use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId};

    /// The process's memory: two pages the guest may read and write, the message buffer and
    /// then the port name.
    const BASE: u64 = 0x8000_0000;
    const PAGES: u64 = 2;
    const TLS: u64 = BASE;
    const NAME: u64 = BASE + PAGE_SIZE;
    const PORT: &str = "gifts";
    const SERVICE: ServiceId = ServiceId(0x71);
    const THREAD: ThreadId = ThreadId(1);
    const CORE: u8 = 0;
    /// The commands of [`Gifts`].
    const EVENT_SESSION_EVENT: u32 = 0;
    const ONE_EVENT: u32 = 1;

    /// An interface that copies the caller handles to events it makes, and a session on itself.
    /// The events it made go to `made`, so the test can count their references.
    #[derive(serde::Serialize)]
    struct Gifts {
        #[serde(skip)]
        made: Arc<Mutex<Vec<ObjectId>>>,
    }

    commands! {
        impl Gifts {
            EVENT_SESSION_EVENT => event_session_event,
            ONE_EVENT => one_event,
        }
    }

    impl Gifts {
        fn event(&self, cx: &mut Context<'_>) -> ObjectId {
            let event = cx.create_event().unwrap();
            self.made.lock().unwrap().push(event);
            event
        }

        /// Copies an event, a new session on this server, and another event, in that order.
        fn event_session_event(
            this: &mut Self,
            cx: &mut Context<'_>,
            _: &Request<'_>,
            reply: &mut Reply,
        ) -> MethodOutcome {
            let (first, second) = (this.event(cx), this.event(cx));
            let made = Arc::clone(&this.made);
            reply
                .copy_handle(first)
                .copy_session(Box::new(Gifts { made }))
                .copy_handle(second);
            reply.ok()
        }

        /// Copies one event.
        fn one_event(
            this: &mut Self,
            cx: &mut Context<'_>,
            _: &Request<'_>,
            reply: &mut Reply,
        ) -> MethodOutcome {
            let event = this.event(cx);
            reply.copy_handle(event);
            reply.ok()
        }
    }

    struct NoDebug;

    impl DebugSink for NoDebug {
        fn output(&mut self, _: GuestTicks, _: &[u8]) {}
    }

    /// A Switch 2 process whose handle table holds `table_size` handles, with the `gifts`
    /// server on a port, and the events its interfaces made.
    struct Rig {
        dispatcher: Dispatcher,
        memory: FlatMemory,
        env: LiveEnv,
        recorder: Recorder,
        made: Arc<Mutex<Vec<ObjectId>>>,
    }

    /// A reply: its result and its copied handles.
    struct Answer {
        result: ResultCode,
        copies: Vec<Handle>,
    }

    impl Rig {
        fn new(table_size: i32) -> Self {
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
            let mut description = ProcessDescription::homebrew(Console::Switch2);
            description.handle_table_size = table_size;
            let tables = process::create(&description, space).unwrap();
            let dispatcher = Dispatcher::new(tables);
            let made = Arc::new(Mutex::new(Vec::new()));
            let root = Arc::clone(&made);
            let server = Server::new(PORT, UnknownPolicy::Strict).with_root(move || {
                Box::new(Gifts {
                    made: Arc::clone(&root),
                })
            });
            dispatcher
                .add_server(
                    PORT,
                    SERVICE,
                    Some(PortName::new(PORT).unwrap()),
                    Box::new(server),
                )
                .unwrap();
            Self {
                dispatcher,
                memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
                env: LiveEnv::new(FixedClock::new(0, 1), 0),
                recorder: Recorder::new(),
                made,
            }
        }

        fn call(&self, id: SvcId, x: [u64; 4]) -> GuestRegs {
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
            assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
            regs
        }

        fn connect(&self) -> Handle {
            let mut name = PORT.as_bytes().to_vec();
            name.push(0);
            self.memory.write(NAME, &name).unwrap();
            let regs = self.call(SvcId::ConnectToNamedPort, [0, NAME, 0, 0]);
            assert_eq!(regs.x[0], 0);
            Handle::from_register(regs.x[1])
        }

        /// Sends `command` on `session`; the system call must succeed. Returns the reply.
        fn send(&self, session: Handle, command: u32) -> Answer {
            let mut buffer = [0; BUFFER_SIZE];
            RequestBuilder::cmif(CommandType::Request, command)
                .build(&mut buffer)
                .unwrap();
            self.memory.write(TLS, &buffer).unwrap();
            let regs = self.call(SvcId::SendSyncRequest, [u64::from(session.raw()), 0, 0, 0]);
            assert_eq!(regs.x[0], 0);
            self.memory.read(TLS, &mut buffer).unwrap();
            let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
            let Body::CmifReply(reply) = &message.body else {
                unreachable!("not a CMIF reply");
            };
            Answer {
                result: ResultCode::from_raw(reply.header.result),
                copies: message
                    .copy_handles
                    .iter()
                    .copied()
                    .map(Handle::from_raw)
                    .collect(),
            }
        }

        /// The process's handle count, the number of objects (the process's own among them),
        /// and the reference count of each event made so far.
        fn holdings(&self) -> (usize, usize, Vec<u32>) {
            let made = self.made.lock().unwrap().clone();
            self.dispatcher
                .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
                    let refs = made
                        .iter()
                        .map(|event| scheduler.objects().refs(*event))
                        .collect();
                    (handles.len(), scheduler.objects().len(), refs)
                })
        }

        /// The `gifts` server's own count of its open sessions and of its live interfaces.
        fn server_counts(&self) -> (usize, usize) {
            let mut counts = Vec::new();
            let parked = self.dispatcher.inspect_handlers(|service, handler| {
                if service == SERVICE {
                    let any: &dyn Any = handler;
                    let server = any.downcast_ref::<Server>().unwrap();
                    counts.push((server.session_count(), server.interface_count()));
                }
            });
            assert_eq!(parked, 0, "a handler is out of its slot");
            assert_eq!(counts.len(), 1, "one gifts server");
            counts[0]
        }
    }

    #[test]
    fn a_reply_the_table_fills_up_part_way_through_leaves_the_caller_as_it_was() {
        // The session's handle and room for two more: the first event and the new session are
        // placed, and the second event does not fit.
        let rig = Rig::new(3);
        let session = rig.connect();
        let before = rig.server_counts();
        let answer = rig.send(session, EVENT_SESSION_EVENT);
        assert_eq!(answer.result, KernelError::OutOfHandles.code());
        assert_eq!(answer.copies, Vec::<Handle>::new());
        // One handle, the session's; the objects are the process, that session and the two
        // events, which only the server holds now (its creator's references). The new session
        // went with its handle.
        assert_eq!(rig.holdings(), (1, 4, vec![1, 1]));
        // The server forgot the session it recorded for the reply, and its interface: its
        // sessions and interfaces are as they were before the request.
        assert_eq!(rig.server_counts(), before);
        // The table has its room back, and a reply that fits keeps what it names.
        let answer = rig.send(session, ONE_EVENT);
        assert_eq!(answer.result, ResultCode::SUCCESS);
        assert_eq!(answer.copies.len(), 1);
        assert_eq!(rig.holdings(), (2, 5, vec![1, 1, 2]));
    }

    #[test]
    fn a_reply_that_fits_keeps_its_handles_and_its_session() {
        // Room for the port's session, the reply's three handles and one more.
        let rig = Rig::new(5);
        let session = rig.connect();
        let (sessions, interfaces) = rig.server_counts();
        let answer = rig.send(session, EVENT_SESSION_EVENT);
        assert_eq!(answer.result, ResultCode::SUCCESS);
        assert_eq!(answer.copies.len(), 3);
        // The server keeps the session it opened for the reply, and that session's interface.
        assert_eq!(rig.server_counts(), (sessions + 1, interfaces + 1));
        // Four handles: the port's session, the two events and the new session, each event
        // held by the server and the caller; and the process's own object.
        assert_eq!(rig.holdings(), (4, 5, vec![2, 2]));
        // The new session works: it answers on the same interface.
        let answer = rig.send(answer.copies[1], ONE_EVENT);
        assert_eq!(answer.result, ResultCode::SUCCESS);
        assert_eq!(rig.holdings(), (5, 6, vec![2, 2, 2]));
    }
}
