//! The graphics server's four kernel calls on the real kernel: a service's [`Context`] reaches
//! the dispatcher's `IpcContext` through the framework's `Kernel` implementation, and a guest's
//! system calls see what the service did (the hooked event's writable end, the GPU-accessible
//! attribute `svc 0x80` sets, and the device-share lock).

/// Everything in a `cfg(test)` module, so that the helpers may unwrap as the tests do.
#[cfg(test)]
mod on_the_kernel {
    use std::sync::{Arc, Mutex, PoisonError};

    use serde::{Deserialize, Serialize};
    use starquake_cpu::GuestRegs;
    use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
    use starquake_kernel::dispatch::{Caller, DeviceShareError, Dispatcher, Outcome};
    use starquake_kernel::handle::Handle;
    use starquake_kernel::id::ObjectId;
    use starquake_kernel::ipc::{EventHook, PortName};
    use starquake_kernel::memory::testing::RecordingMapper;
    use starquake_kernel::memory::{
        AddressSpace, DeviceShareToken, MemoryArrangement, MemoryAttribute, ProcessMemory,
        RegionOffsets, SvcPermission,
    };
    use starquake_kernel::process::{self, Console, ProcessDescription};
    use starquake_kernel::result::{KernelError, ResultCode};
    use starquake_kernel::svc::{DebugSink, SvcId};
    use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
    use starquake_services::commands;
    use starquake_services::framework::{
        Context, Fault, Outcome as Answer, Reply, Request, Server, UnknownPolicy, Writer,
    };
    use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ServiceId, ThreadId};

    /// The process's memory: four pages the guest may read and write. The first is the message
    /// buffer, the second holds the port name, and the last two are what the service works on.
    const BASE: u64 = 0x8000_0000;
    const PAGES: u64 = 4;
    const TLS: u64 = BASE;
    const NAME: u64 = BASE + PAGE_SIZE;
    const WORK: u64 = BASE + 2 * PAGE_SIZE;

    /// The commands. `MAKE_EVENT`: out a copy handle. `QUERY`: in `u64 address, u64 size`, out
    /// `u8`. `LOCK`, `UNLOCK`: in `u64 address, u64 size`.
    const MAKE_EVENT: u32 = 0;
    const QUERY: u32 = 1;
    const LOCK: u32 = 2;
    const UNLOCK: u32 = 3;

    struct NoDebug;

    impl DebugSink for NoDebug {
        fn output(&mut self, _: GuestTicks, _: &[u8]) {}
    }

    /// What a hook heard.
    #[derive(Default, Clone)]
    struct Heard(Arc<Mutex<Vec<ObjectId>>>);

    impl Heard {
        fn events(&self) -> Vec<ObjectId> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl EventHook for Heard {
        fn signalled(&self, event: ObjectId) {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event);
        }
    }

    /// The tokens a service's locks returned, oldest first, shared with the test.
    #[derive(Default, Clone)]
    struct Tokens(Arc<Mutex<Vec<DeviceShareToken>>>);

    impl Tokens {
        fn all(&self) -> Vec<DeviceShareToken> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    /// The service: one session's worth of the four calls. Its hook and its tokens are not part
    /// of a snapshot.
    #[derive(Default, Serialize, Deserialize)]
    struct Gpu {
        #[serde(skip)]
        heard: Heard,
        #[serde(skip)]
        tokens: Tokens,
    }

    commands! {
        impl Gpu {
            MAKE_EVENT => make_event,
            QUERY => query,
            LOCK => lock,
            UNLOCK => unlock,
        }
    }

    fn range(req: &Request<'_>) -> Result<(u64, u64), Fault> {
        let mut payload = req.payload();
        Ok((payload.read_u64()?, payload.read_u64()?))
    }

    impl Gpu {
        fn make_event(
            &mut self,
            cx: &mut Context<'_>,
            _: &Request<'_>,
            reply: &mut Reply,
        ) -> Answer {
            let hook: Arc<dyn EventHook> = Arc::new(self.heard.clone());
            let (writable, _readable) = cx.create_hooked_event(hook)?;
            reply.copy_handle(writable);
            reply.ok()
        }

        fn query(
            _: &mut Self,
            cx: &mut Context<'_>,
            req: &Request<'_>,
            reply: &mut Reply,
        ) -> Answer {
            let (address, size) = range(req)?;
            let accessible = cx.is_gpu_accessible(address, size)?;
            reply.payload().write_u8(u8::from(accessible));
            reply.ok()
        }

        fn lock(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Answer {
            let (address, size) = range(req)?;
            let token = cx.lock_device_share(address, size)?;
            self.tokens
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(token);
            reply.ok()
        }

        /// Unlocks with the latest token a lock of the request's range returned, which stays
        /// held, so a second unlock presents it again.
        fn unlock(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Answer {
            let (address, size) = range(req)?;
            let token = self
                .tokens
                .all()
                .into_iter()
                .rev()
                .find(|token| (token.address(), token.size()) == (address, size))
                .expect("a lock of the range");
            cx.unlock_device_share(token)?;
            reply.ok()
        }
    }

    /// A Switch 2 process with its four pages read-write, the service on the port `gpu:`, and the
    /// guest's memory.
    struct Rig {
        dispatcher: Dispatcher,
        memory: FlatMemory,
        env: LiveEnv,
        recorder: Recorder,
    }

    impl Rig {
        fn new(heard: &Heard, tokens: &Tokens) -> Self {
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
            let dispatcher = Dispatcher::new(process::create(&description, space).unwrap());
            let heard = heard.clone();
            let tokens = tokens.clone();
            let server = Server::new("gpu", UnknownPolicy::Strict)
                .with_service_id(ServiceId(0x62))
                .with_root(move || {
                    Box::new(Gpu {
                        heard: heard.clone(),
                        tokens: tokens.clone(),
                    })
                });
            dispatcher
                .add_server(
                    "gpu",
                    ServiceId(0x62),
                    Some(PortName::new("gpu:").unwrap()),
                    Box::new(server),
                )
                .unwrap();
            Self {
                dispatcher,
                memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
                env: LiveEnv::new(FixedClock::new(0, 1), 0),
                recorder: Recorder::new(),
            }
        }

        /// Makes system call `id` with `x0` to `x3`; returns the registers after it.
        fn call(&self, id: SvcId, x: [u64; 4]) -> GuestRegs {
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

        fn connect(&self) -> Handle {
            self.memory.write(NAME, b"gpu:\0").unwrap();
            let regs = self.call(SvcId::ConnectToNamedPort, [0, NAME, 0, 0]);
            assert_eq!(regs.x[0], 0);
            Handle::from_register(regs.x[1])
        }

        /// Sends `request` on `session`: the reply's result, copy handles and data.
        fn send(&self, session: Handle, request: &RequestBuilder) -> (u32, Vec<u32>, Vec<u8>) {
            let mut buffer = [0; BUFFER_SIZE];
            request.build(&mut buffer).unwrap();
            self.memory.write(TLS, &buffer).unwrap();
            let regs = self.call(SvcId::SendSyncRequest, [u64::from(session.raw()), 0, 0, 0]);
            assert_eq!(regs.x[0], 0);
            self.memory.read(TLS, &mut buffer).unwrap();
            let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
            let Body::CmifReply(reply) = &message.body else {
                unreachable!("not a CMIF reply");
            };
            (
                reply.header.result,
                message.copy_handles.clone(),
                reply.data.to_vec(),
            )
        }
    }

    fn with_range(command: u32, address: u64, size: u64) -> RequestBuilder {
        let mut writer = Writer::new();
        writer.write_u64(address).write_u64(size);
        RequestBuilder::cmif(CommandType::Request, command).data(writer.as_bytes())
    }

    fn code(error: KernelError) -> u32 {
        error.code().raw()
    }

    #[test]
    fn the_guest_signals_the_writable_end_a_service_made_and_its_hook_hears() {
        let heard = Heard::default();
        let rig = Rig::new(&heard, &Tokens::default());
        let session = rig.connect();
        let (result, copies, _) = rig.send(
            session,
            &RequestBuilder::cmif(CommandType::Request, MAKE_EVENT),
        );
        assert_eq!(result, ResultCode::SUCCESS.raw());
        let writable = Handle::from_raw(copies[0]);
        assert_eq!(heard.events(), []);
        let regs = rig.call(SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
        assert_eq!(regs.x[0], 0);
        assert_eq!(heard.events().len(), 1);
        // A clear runs nothing.
        let regs = rig.call(SvcId::ClearEvent, [u64::from(writable.raw()), 0, 0, 0]);
        assert_eq!(regs.x[0], 0);
        assert_eq!(heard.events().len(), 1);
    }

    #[test]
    fn a_service_sees_the_attribute_the_guest_set_and_locks_its_memory() {
        let heard = Heard::default();
        let rig = Rig::new(&heard, &Tokens::default());
        let session = rig.connect();
        let query = |rig: &Rig| rig.send(session, &with_range(QUERY, WORK, PAGE_SIZE)).2[0];
        assert_eq!(query(&rig), 0);
        // The guest marks the page with `svc 0x80`.
        let gpu = u64::from(MemoryAttribute::GPU_ACCESSIBLE.svc());
        let regs = rig.call(SvcId::SetGpuMemoryAttribute, [WORK, PAGE_SIZE, gpu, gpu]);
        assert_eq!(regs.x[0], 0);
        assert_eq!(query(&rig), 1);
        // The next page, locked, cannot be reprotected by the guest; unlocked, it can.
        let other = WORK + PAGE_SIZE;
        let read = u64::from(SvcPermission::Read.bits());
        let reprotect =
            |rig: &Rig| rig.call(SvcId::SetMemoryPermission, [other, PAGE_SIZE, read, 0]);
        assert_eq!(rig.send(session, &with_range(LOCK, other, PAGE_SIZE)).0, 0);
        assert_eq!(
            reprotect(&rig).x[0],
            u64::from(code(KernelError::InvalidCurrentMemory))
        );
        assert_eq!(
            rig.send(session, &with_range(UNLOCK, other, PAGE_SIZE)).0,
            0
        );
        assert_eq!(reprotect(&rig).x[0], 0);
        // A second unlock is the kernel's refusal, as the reply's result.
        assert_eq!(
            rig.send(session, &with_range(UNLOCK, other, PAGE_SIZE)).0,
            code(KernelError::InvalidCurrentMemory)
        );
    }

    #[test]
    fn a_lock_a_service_took_is_released_by_its_token_outside_any_request() {
        // The timer thread's path: the token leaves the request, and the dispatcher unlocks it
        // with no guest thread and no request (graphics-server spec, "The model").
        let tokens = Tokens::default();
        let rig = Rig::new(&Heard::default(), &tokens);
        let session = rig.connect();
        let read = u64::from(SvcPermission::Read.bits());
        let reprotect =
            |rig: &Rig| rig.call(SvcId::SetMemoryPermission, [WORK, PAGE_SIZE, read, 0]);
        assert_eq!(rig.send(session, &with_range(LOCK, WORK, PAGE_SIZE)).0, 0);
        let all = tokens.all();
        assert_eq!(all.len(), 1);
        let token = all[0];
        assert_eq!((token.address(), token.size()), (WORK, PAGE_SIZE));
        assert_eq!(
            reprotect(&rig).x[0],
            u64::from(code(KernelError::InvalidCurrentMemory))
        );
        assert_eq!(rig.dispatcher.unlock_device_share(token), Ok(()));
        assert_eq!(reprotect(&rig).x[0], 0);
        // Its lock is gone: the token is refused, on the timer thread's path and in a request.
        assert_eq!(
            rig.dispatcher.unlock_device_share(token),
            Err(DeviceShareError::NotLocked {
                address: WORK,
                size: PAGE_SIZE,
            })
        );
        assert_eq!(
            rig.send(session, &with_range(UNLOCK, WORK, PAGE_SIZE)).0,
            code(KernelError::InvalidCurrentMemory)
        );
    }
}
