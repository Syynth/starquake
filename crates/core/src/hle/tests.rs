//! The run's services behind the real dispatcher, on every host: the conversation the homebrew
//! fixture has with them, request by request, as libnx lays its requests out.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use starquake_cpu::GuestRegs;
use starquake_ipc::{
    BUFFER_SIZE, BufferDescriptor, CommandType, Message, ReceiveDescriptor, RequestBuilder,
    RequestLayout, SessionKind, StaticDescriptor,
};
use starquake_kernel::dispatch::{Caller, Outcome};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ServerId;
use starquake_kernel::memory::testing::RecordingMapper;
use starquake_kernel::memory::{
    AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets, SvcPermission,
};
use starquake_kernel::process::{self, Console, ProcessDescription};
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_kernel::svc::{DebugSink, SvcId};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_services::framework::control::{
    CONVERT_CURRENT_OBJECT_TO_DOMAIN, QUERY_POINTER_BUFFER_SIZE,
};
use starquake_services::framework::{DEFAULT_POINTER_BUFFER_SIZE, Protocol, Session};
use starquake_services::fsp::halt::{Halt, Stop};
use starquake_services::fsp::worker::{self, Recording, Workers};
use starquake_services::fsp::{FspSrv, IFile, IFileSystem};
use starquake_services::set::{self, FIRMWARE_VERSION_SIZE, FirmwareVersion, SetSys, layout};
use starquake_services::sm::{MAX_NAME_LEN, ServiceName, Sm};
use starquake_trace::{EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId, VecSink};

use super::*;

mod data_file_system;
mod data_storage;

/// Where the process's memory is: four pages the guest may read and write.
const BASE: u64 = 0x8000_0000;
const PAGES: u64 = 4;
/// The thread's message buffer, the port name, and the buffer `GetFirmwareVersion` fills.
const TLS: u64 = BASE;
const NAME: u64 = BASE + PAGE_SIZE;
const FIRMWARE: u64 = BASE + 2 * PAGE_SIZE;

const THREAD: ThreadId = ThreadId(1);

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// The run's services behind a dispatcher, over memory the guest may read and write.
struct Rig {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: Arc<LiveEnv>,
    recorder: Recorder,
    events: Arc<VecSink>,
    ready: Arc<ReadyServers>,
}

/// The longest a test waits for a read to finish.
const WAIT: Duration = Duration::from_secs(10);

/// The servers whose deferred replies are ready, as the core loop's `Ready` keeps them.
#[derive(Default)]
struct ReadyServers {
    servers: Mutex<Vec<ServerId>>,
    rung: Condvar,
}

impl Deliveries for ReadyServers {
    fn ready(&self, server: ServerId) {
        self.servers.lock().unwrap().push(server);
        self.rung.notify_all();
    }
}

impl ReadyServers {
    /// Waits until a server is named, and takes the oldest.
    fn next(&self) -> ServerId {
        let servers = self.servers.lock().unwrap();
        let (mut servers, timeout) = self
            .rung
            .wait_timeout_while(servers, WAIT, |servers| servers.is_empty())
            .unwrap();
        assert!(!timeout.timed_out(), "no reply became ready");
        servers.remove(0)
    }
}

/// A halt for services whose test expects the environment never to fail: a stop fails the test.
struct Unexpected;

impl Halt for Unexpected {
    fn halt(&self, stop: Stop) {
        panic!("the run was told to stop: {stop}");
    }
}

/// `fsp-srv`'s reads on its workers, over `env`, telling `ready`.
fn reads_on(env: Arc<dyn Env>, ready: &Arc<ReadyServers>) -> Reads {
    let halt: Arc<dyn Halt> = Arc::new(Unexpected);
    let workers = Workers::new(
        vec![env; worker::COUNT.get()],
        Recording::Off,
        Arc::clone(&halt),
    )
    .unwrap();
    let ready: Arc<dyn Deliveries> = Arc::<ReadyServers>::clone(ready);
    Reads {
        executor: Arc::new(workers),
        ready,
        halt,
    }
}

/// [`install`] with reads that nothing waits for.
fn install_with(dispatcher: &Dispatcher, env: Arc<dyn Env>) -> Result<(), ServicesError> {
    let reads = reads_on(Arc::clone(&env), &Arc::default());
    install(dispatcher, env, reads)
}

/// A dispatcher over a process with nothing but its code mapped, read-write.
fn dispatcher() -> Dispatcher {
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
    let description = ProcessDescription::homebrew(Console::Switch1);
    Dispatcher::new(process::create(&description, space).unwrap())
}

impl Rig {
    fn new() -> Self {
        Self::with_env(LiveEnv::new(FixedClock::new(0, 1), 0))
    }

    /// A rig whose kernel and services read `env`.
    fn with_env(env: LiveEnv) -> Self {
        let dispatcher = dispatcher();
        let env = Arc::new(env);
        let services: Arc<dyn Env> = Arc::<LiveEnv>::clone(&env);
        let ready = Arc::new(ReadyServers::default());
        let fsp_reads = reads_on(Arc::clone(&services), &ready);
        install(&dispatcher, services, fsp_reads).unwrap();
        let recorder = Recorder::new();
        let events = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&events));
        Self {
            dispatcher,
            memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
            env,
            recorder,
            events,
            ready,
        }
    }

    fn call(&self, id: SvcId, x0: u64, x1: u64) -> GuestRegs {
        let (regs, outcome) = self.call_for(id, x0, x1);
        assert_eq!(outcome, Outcome::Completed, "{}", id.name());
        regs
    }

    /// Makes the call and returns its registers and outcome.
    fn call_for(&self, id: SvcId, x0: u64, x1: u64) -> (GuestRegs, Outcome) {
        let mut regs = GuestRegs {
            tpidrro: TLS,
            ..GuestRegs::default()
        };
        regs.x[0] = x0;
        regs.x[1] = x1;
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env: &*self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut caller);
        (regs, record.outcome)
    }

    /// Sends `request` on `session`, which defers it: waits for the reply to be ready, has the
    /// dispatcher deliver it, as the core loop's timer thread does, and returns the result the
    /// sender wakes with and its message buffer.
    fn send_deferred(&self, session: Handle, request: &RequestBuilder) -> (ResultCode, Vec<u8>) {
        let mut buffer = [0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.memory.write(TLS, &buffer).unwrap();
        let (_, outcome) = self.call_for(SvcId::SendSyncRequest, u64::from(session.raw()), 0);
        assert_eq!(outcome, Outcome::Waiting);
        let server = self.ready.next();
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env: &*self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        self.dispatcher
            .deliver_replies(server, &mut caller)
            .unwrap();
        let woken = self.dispatcher.take_woken();
        assert_eq!(woken.len(), 1);
        self.memory.read(TLS, &mut buffer).unwrap();
        (woken[0].result, buffer.to_vec())
    }

    /// Sends `request` on `session`; returns the result and the message buffer afterwards.
    fn send(&self, session: Handle, request: &RequestBuilder) -> (ResultCode, Vec<u8>) {
        let mut buffer = [0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.memory.write(TLS, &buffer).unwrap();
        let regs = self.call(SvcId::SendSyncRequest, u64::from(session.raw()), 0);
        self.memory.read(TLS, &mut buffer).unwrap();
        (result(&regs), buffer.to_vec())
    }

    /// The IPC events so far: each one's service and command.
    fn ipc(&self) -> Vec<(ServiceId, u32)> {
        let events = self.events.events();
        events
            .iter()
            .filter_map(|event| match event.kind {
                EventKind::Ipc {
                    service, command, ..
                } => Some((service, command)),
                _ => None,
            })
            .collect()
    }
}

/// A folder for one test, unique to it and to the test process, removed on drop. The SD card is
/// its `sd` subfolder ([`Scratch::path`]), so a test can put files beside the card, outside it,
/// and still inside the scratch folder.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(test: &str) -> Self {
        let name = format!("starquake-sd-{}-{test}", std::process::id());
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sd")).unwrap();
        Self(dir)
    }

    /// The SD card's folder.
    pub(crate) fn path(&self) -> PathBuf {
        self.0.join("sd")
    }

    /// Writes a file at `path`, relative to the SD card's folder.
    pub(crate) fn file(&self, path: &str, bytes: &[u8]) {
        let path = self.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// An environment with no host I/O: a run with no SD card folder.
fn no_host() -> Arc<dyn Env> {
    Arc::new(LiveEnv::new(FixedClock::new(0, 1), 0))
}

fn result(regs: &GuestRegs) -> ResultCode {
    ResultCode::from_raw(u32::try_from(regs.x[0]).unwrap())
}

/// The id a command is listed under in an interface's table, by its method's name.
fn command(interface: &dyn Session, name: &str) -> u32 {
    let table = interface.commands();
    let found = table.cmif.iter().find(|command| command.name == name);
    found.map(|command| command.id).unwrap()
}

/// A request laid out as libnx lays it out (`cmifMakeRequest`).
fn libnx(id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, id).layout(RequestLayout::Libnx)
}

/// The reply's result and data, read as libnx reads a reply of `size` bytes.
fn reply(buffer: &[u8], size: usize) -> (u32, Vec<u8>, Vec<u32>) {
    let message = Message::parse_reply_sized(buffer, SessionKind::Session, size).unwrap();
    match message.body {
        starquake_ipc::Body::CmifReply(body) => {
            (body.header.result, body.data.to_vec(), message.move_handles)
        }
        other => unreachable!("not a CMIF reply: {other:?}"),
    }
}

#[test]
fn every_service_has_its_name_and_no_other_number_has_one() {
    assert_eq!(service_name(SM), Some("sm:"));
    assert_eq!(service_name(SET_SYS), Some("set:sys"));
    assert_eq!(service_name(FSP_SRV), Some("fsp-srv"));
    assert_eq!(service_name(APM), Some("apm"));
    assert_eq!(service_name(APPLET_OE), Some("appletOE"));
    assert_eq!(service_name(HID), Some("hid"));
    assert_eq!(service_name(TIME_U), Some("time:u"));
    assert_eq!(service_name(GSV_ROOT), Some("gsv"));
    assert_eq!(service_name(ServiceId(0)), None);
    assert_eq!(service_name(ServiceId(9)), None);
}

#[test]
fn the_service_manager_is_on_its_port_and_set_sys_on_none() {
    let dispatcher = dispatcher();
    install_with(&dispatcher, no_host()).unwrap();
    let tables = dispatcher.into_tables();
    let ipc = tables.scheduler.ipc();
    let sm = ipc.port(&PortName::new(sm::PORT_NAME).unwrap()).unwrap();
    assert_eq!(ipc.service(SM), Some(sm));
    assert_eq!(ipc.server_name(sm), Ok(sm::PORT_NAME));
    let set = ipc.service(SET_SYS).unwrap();
    assert_eq!(ipc.server_name(set), Ok(set::SYS_NAME));
    assert_eq!(ipc.port(&PortName::new(set::SYS_NAME).unwrap()), None);
    let fsp = ipc.service(FSP_SRV).unwrap();
    assert_eq!(ipc.server_name(fsp), Ok(fsp::NAME));
    assert_eq!(ipc.port(&PortName::new(fsp::NAME).unwrap()), None);
}

#[test]
fn installing_twice_is_refused_by_the_kernel() {
    let dispatcher = dispatcher();
    install_with(&dispatcher, no_host()).unwrap();
    let port = PortName::new(sm::PORT_NAME).unwrap();
    assert_eq!(
        install_with(&dispatcher, no_host()),
        Err(ServicesError::Kernel(IpcError::PortTaken(port)))
    );
}

/// The fixture's conversation (`tests/homebrew/hello/hello.c`, through libnx's `smInitialize`,
/// `setsysInitialize`, `setsysGetFirmwareVersion` and the two `serviceClose`s), request by
/// request.
#[test]
fn the_fixtures_conversation_with_sm_and_set_sys() {
    let rig = Rig::new();
    let sm_interface = Sm::default();
    let set_interface = SetSys;

    rig.memory.write(NAME, b"sm:\0").unwrap();
    let regs = rig.call(SvcId::ConnectToNamedPort, 0, NAME);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let sm = Handle::from_register(regs.x[1]);

    // libnx asks every new session for its pointer buffer size first.
    let query = RequestBuilder::cmif(CommandType::Control, QUERY_POINTER_BUFFER_SIZE)
        .layout(RequestLayout::Libnx);
    let (code, buffer) = rig.send(sm, &query);
    assert_eq!(code, ResultCode::SUCCESS);
    let (word, data, _) = reply(&buffer, size_of::<u16>());
    assert_eq!(word, 0);
    assert_eq!(data, DEFAULT_POINTER_BUFFER_SIZE.to_le_bytes());

    let initialize = libnx(command(&sm_interface, "initialize"))
        .send_pid(0)
        .data(&0_u64.to_le_bytes());
    let (code, buffer) = rig.send(sm, &initialize);
    assert_eq!((code, reply(&buffer, 0).0), (ResultCode::SUCCESS, 0));

    let name: [u8; MAX_NAME_LEN] = ServiceName::new(set::SYS_NAME).unwrap().to_field();
    let get_service = libnx(command(&sm_interface, "get_service")).data(&name);
    let (code, buffer) = rig.send(sm, &get_service);
    let (word, _, moved) = reply(&buffer, 0);
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    let set_sys = Handle::from_raw(moved[0]);

    let (code, _) = rig.send(set_sys, &query);
    assert_eq!(code, ResultCode::SUCCESS);
    let room = ReceiveDescriptor {
        address: FIRMWARE,
        size: u16::try_from(FIRMWARE_VERSION_SIZE).unwrap(),
    };
    let get_firmware =
        libnx(command(&set_interface, "get_firmware_version")).receive_descriptor(room);
    let (code, buffer) = rig.send(set_sys, &get_firmware);
    assert_eq!((code, reply(&buffer, 0).0), (ResultCode::SUCCESS, 0));
    let mut written = [0; FIRMWARE_VERSION_SIZE];
    rig.memory.read(FIRMWARE, &mut written).unwrap();
    let mut expected = FirmwareVersion::eden_default().to_bytes();
    expected[layout::REVISION_MINOR] = 0;
    assert_eq!(written, expected);

    // `serviceClose`: a close request, which ends with the session closed, then the handle.
    let closed = KernelError::SessionClosed.code();
    assert_eq!(rig.send(set_sys, &RequestBuilder::close()).0, closed);
    assert_eq!(
        result(&rig.call(SvcId::CloseHandle, u64::from(set_sys.raw()), 0)),
        ResultCode::SUCCESS
    );
    assert_eq!(rig.send(sm, &RequestBuilder::close()).0, closed);
    assert_eq!(
        result(&rig.call(SvcId::CloseHandle, u64::from(sm.raw()), 0)),
        ResultCode::SUCCESS
    );

    // One IPC event per answered request, none for the closes.
    let get_firmware_id = command(&set_interface, "get_firmware_version");
    assert_eq!(
        rig.ipc(),
        [
            (SM, QUERY_POINTER_BUFFER_SIZE),
            (SM, command(&sm_interface, "initialize")),
            (SM, command(&sm_interface, "get_service")),
            (SET_SYS, QUERY_POINTER_BUFFER_SIZE),
            (SET_SYS, get_firmware_id),
        ]
    );
    // Every session and handle is gone: the one object left is the process's own.
    let tables = rig.dispatcher.into_tables();
    assert!(tables.handles.is_empty());
    let own = tables.processes.own();
    let process = tables.scheduler.process(own).unwrap().object();
    assert_eq!(tables.scheduler.objects().len(), 1);
    assert_eq!(
        tables.scheduler.objects().get(process),
        Ok(starquake_kernel::object::Object::Process(own))
    );
    let names = set_interface.commands();
    assert_eq!(
        names.name(Protocol::Cmif, get_firmware_id),
        Some("get_firmware_version")
    );
}

/// Where the caller keeps a path for `OpenFile`.
const PATH: u64 = BASE + 3 * PAGE_SIZE;

/// A domain reply's result, data and object ids, read as libnx reads a reply of `size` bytes.
fn domain_reply(buffer: &[u8], size: usize) -> (u32, Vec<u8>, Vec<u32>) {
    let message = Message::parse_reply_sized(buffer, SessionKind::Domain, size).unwrap();
    match message.body {
        starquake_ipc::Body::CmifReply(body) => {
            (body.header.result, body.data.to_vec(), body.object_ids)
        }
        other => unreachable!("not a CMIF reply: {other:?}"),
    }
}

impl Rig {
    /// A session on `name`, from the service manager, as libnx gets one (`smGetService`).
    fn service(&self, name: &str) -> Handle {
        self.memory.write(NAME, b"sm:\0").unwrap();
        let regs = self.call(SvcId::ConnectToNamedPort, 0, NAME);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
        let sm = Handle::from_register(regs.x[1]);
        let sm_interface = Sm::default();
        let initialize = libnx(command(&sm_interface, "initialize"))
            .send_pid(0)
            .data(&0_u64.to_le_bytes());
        assert_eq!(self.send(sm, &initialize).0, ResultCode::SUCCESS);
        let name: [u8; MAX_NAME_LEN] = ServiceName::new(name).unwrap().to_field();
        let get_service = libnx(command(&sm_interface, "get_service")).data(&name);
        let (code, buffer) = self.send(sm, &get_service);
        let (word, _, moved) = reply(&buffer, 0);
        assert_eq!((code, word), (ResultCode::SUCCESS, 0));
        Handle::from_raw(moved[0])
    }

    /// Makes `session` a domain, as libnx does (`serviceConvertToDomain`), and returns the
    /// root's object id.
    fn convert(&self, session: Handle) -> u32 {
        let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN)
            .layout(RequestLayout::Libnx);
        let (code, buffer) = self.send(session, &convert);
        assert_eq!(code, ResultCode::SUCCESS);
        let (word, data, _) = reply(&buffer, size_of::<u32>());
        assert_eq!(word, 0);
        u32::from_le_bytes(data.try_into().unwrap())
    }
}

/// A request to object `object` of a domain, for the command `name` of `interface`.
fn domain(object: u32, interface: &dyn Session, name: &str) -> RequestBuilder {
    RequestBuilder::domain(CommandType::Request, object, command(interface, name))
}

/// `fsp-srv` as libnx starts it (`_fsInitialize`, then `fsdevMountSdmc`), and a file opened on
/// the SD card the run names, through the real dispatcher.
#[test]
fn fsp_srv_serves_the_run_s_sd_card_folder() {
    const TEXT: &[u8] = b"a file on the card";
    let scratch = Scratch::new("hle-fsp");
    scratch.file("hello.txt", TEXT);
    let folder = HostFolder::new(scratch.path());
    let rig = Rig::with_env(LiveEnv::new(FixedClock::new(0, 1), 0).with_host_io(folder));
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);

    let set = domain(root, &FspSrv, "set_current_process")
        .send_pid(0)
        .data(&0_u64.to_le_bytes());
    let (code, buffer) = rig.send(fsp, &set);
    assert_eq!((code, domain_reply(&buffer, 0).0), (ResultCode::SUCCESS, 0));

    let open_sd = domain(root, &FspSrv, "open_sd_card_file_system");
    let (code, buffer) = rig.send(fsp, &open_sd);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let fs = objects[0];

    let mut path = b"/hello.txt".to_vec();
    path.resize(fsp::PATH_BUFFER_SIZE, 0);
    rig.memory.write(PATH, &path).unwrap();
    let descriptor = StaticDescriptor {
        counter: 0,
        address: PATH,
        size: u16::try_from(path.len()).unwrap(),
    };
    let open_file = domain(fs, &IFileSystem::default(), "open_file")
        .data(&fsp::open_mode::READ.to_le_bytes())
        .static_descriptor(descriptor);
    let (code, buffer) = rig.send(fsp, &open_file);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let file = objects[0];

    let get_size = domain(file, &IFile::default(), "get_size");
    let (code, buffer) = rig.send(fsp, &get_size);
    let (word, data, _) = domain_reply(&buffer, size_of::<i64>());
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    assert_eq!(data, u64::try_from(TEXT.len()).unwrap().to_le_bytes());

    // A read is deferred to `fsp-srv`'s workers, and the bytes arrive with the reply.
    let buffer_room = BufferDescriptor {
        address: FIRMWARE,
        size: 4,
        flags: 0,
    };
    let payload = [
        &0_u32.to_le_bytes()[..],
        &[0; 4],
        &2_i64.to_le_bytes(),
        &4_i64.to_le_bytes(),
    ]
    .concat();
    let read = domain(file, &IFile::default(), "read")
        .data(&payload)
        .receive_buffer(buffer_room);
    let (code, buffer) = rig.send_deferred(fsp, &read);
    let (word, data, _) = domain_reply(&buffer, size_of::<i64>());
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    assert_eq!(data, 4_i64.to_le_bytes());
    let mut read_bytes = [0; 4];
    rig.memory.read(FIRMWARE, &mut read_bytes).unwrap();
    assert_eq!(read_bytes, TEXT[2..6]);

    // One event for each request to `fsp-srv`: the conversion and the five commands.
    let fsp_events = rig
        .ipc()
        .into_iter()
        .filter(|(service, _)| *service == FSP_SRV);
    assert_eq!(fsp_events.count(), 6);
}

#[test]
fn a_run_without_an_sd_card_folder_has_no_sd_card() {
    let rig = Rig::new();
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);
    let open_sd = domain(root, &FspSrv, "open_sd_card_file_system");
    let (code, buffer) = rig.send(fsp, &open_sd);
    assert_eq!(code, ResultCode::SUCCESS);
    let (word, _, objects) = domain_reply(&buffer, 0);
    let no_device = fsp::result::PORT_SD_CARD_NO_DEVICE.raw();
    assert_eq!((word, objects.len()), (no_device, 0));
}

/// `apm` as libnx starts it (`_apmInitialize`): `OpenSession` hands out a session handle.
#[test]
fn apm_hands_out_a_session() {
    let rig = Rig::new();
    let apm_session = rig.service(apm::NAME);
    let open_session = command(&apm::Apm, "open_session");
    let (code, buffer) = rig.send(apm_session, &libnx(open_session));
    let (word, _, moved) = reply(&buffer, 0);
    assert_eq!((code, word, moved.len()), (ResultCode::SUCCESS, 0, 1));
    assert_eq!(rig.ipc().last(), Some(&(APM, open_session)));
}

/// `OpenApplicationProxy` as libnx sends it (`_appletGetSessionProxy`), with `process` copied.
fn open_proxy(root: u32, process: Handle) -> RequestBuilder {
    domain(
        root,
        &am::IApplicationProxyService,
        "open_application_proxy",
    )
    .send_pid(0)
    .copy_handle(process.raw())
    .data(&0_u64.to_le_bytes())
}

/// Waits on `handle` with a zero timeout: the result says whether it was signalled.
fn poll(rig: &Rig, handle: Handle) -> ResultCode {
    rig.memory.write(PATH, &handle.raw().to_le_bytes()).unwrap();
    let mut regs = GuestRegs {
        tpidrro: TLS,
        ..GuestRegs::default()
    };
    // `svcWaitSynchronization(&index, handles, count, timeout)`.
    regs.x[1] = PATH;
    regs.x[2] = 1;
    regs.x[3] = 0;
    let mut debug = NoDebug;
    let mut caller = Caller {
        core: 0,
        thread: THREAD,
        env: &*rig.env,
        memory: &rig.memory,
        debug: &mut debug,
        recorder: &rig.recorder,
    };
    let number = SvcId::WaitSynchronization.number();
    let record = rig.dispatcher.dispatch(number, &mut regs, &mut caller);
    assert_eq!(record.outcome, Outcome::Completed);
    result(&regs)
}

/// `appletOE` as libnx starts it (`_appletInitialize`) through the real dispatcher: the proxy,
/// the two events it returns, which are real kernel events (the message event signalled, the
/// suspended-tick event not), and values the console reports.
#[test]
fn applet_oe_serves_the_start_up_with_real_events() {
    let rig = Rig::new();
    let oe = rig.service(am::OE_NAME);
    let root = rig.convert(oe);

    // libnx copies the current-process pseudo-handle, which the kernel resolves to the calling
    // process (`starquake_kernel::ipc::message`): the service opens a proxy.
    let (code, buffer) = rig.send(oe, &open_proxy(root, Handle::CURRENT_PROCESS));
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));

    // Any handle the caller holds stands in for the process: the service does not use it.
    let (code, buffer) = rig.send(oe, &open_proxy(root, oe));
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let proxy_interface = am::IApplicationProxy::default();
    let open = |name: &str| {
        let (code, buffer) = rig.send(oe, &domain(objects[0], &proxy_interface, name));
        let (word, _, opened) = domain_reply(&buffer, 0);
        assert_eq!(
            (code, word, opened.len()),
            (ResultCode::SUCCESS, 0, 1),
            "{name}"
        );
        opened[0]
    };
    let getter = open("get_common_state_getter");
    let controller = open("get_self_controller");

    let copied = |object: u32, interface: &dyn Session, name: &str| {
        let (code, buffer) = rig.send(oe, &domain(object, interface, name));
        let message = Message::parse_reply_sized(&buffer, SessionKind::Domain, 0).unwrap();
        assert_eq!(code, ResultCode::SUCCESS, "{name}");
        Handle::from_raw(message.copy_handles[0])
    };
    let tick = copied(
        controller,
        &am::ISelfController,
        "get_accumulated_suspended_tick_changed_event",
    );
    let message = copied(getter, &am::ICommonStateGetter, "get_event_handle");
    assert_eq!(poll(&rig, message), ResultCode::SUCCESS);
    assert_eq!(poll(&rig, tick), KernelError::TimedOut.code());

    let asked = [
        ("get_current_focus_state", am::FocusState::InFocus.raw()),
        ("get_operation_mode", am::OperationMode::Docked.raw()),
    ];
    for (name, value) in asked {
        let (code, buffer) = rig.send(oe, &domain(getter, &am::ICommonStateGetter, name));
        let (word, data, _) = domain_reply(&buffer, size_of::<u8>());
        assert_eq!(
            (code, word, data),
            (ResultCode::SUCCESS, 0, vec![value]),
            "{name}"
        );
    }
}

/// The graphics server's root through the real dispatcher: `appletOE`'s `ISelfController` 2000
/// copies a handle to a session on the root's server, which is not on a port, and the title's
/// first TIPC request on it, the bootstrap's `Initialize`, laid out as Eden PR #4588's
/// `DecodeBootstrap` reads it (`eden-pr4588/.../graphics_supervisor_protocol.h` @ e5fb3b97), is
/// answered with success.
#[test]
fn applet_oe_hands_out_the_graphics_server_s_root() {
    use starquake_gsv::protocol::{bootstrap, command};
    use starquake_ipc::TIPC_COMMAND_OFFSET;

    let rig = Rig::new();
    let oe = rig.service(am::OE_NAME);
    let root = rig.convert(oe);
    let (code, buffer) = rig.send(oe, &open_proxy(root, Handle::CURRENT_PROCESS));
    let (_, _, objects) = domain_reply(&buffer, 0);
    assert_eq!(code, ResultCode::SUCCESS);
    let proxy_interface = am::IApplicationProxy::default();
    let (_, buffer) = rig.send(
        oe,
        &domain(objects[0], &proxy_interface, "get_self_controller"),
    );
    let (_, _, controller) = domain_reply(&buffer, 0);
    let ask = domain(
        controller[0],
        &am::ISelfController,
        "get_graphics_supervisor",
    );
    let (code, buffer) = rig.send(oe, &ask);
    assert_eq!(code, ResultCode::SUCCESS);
    let message = Message::parse_reply_sized(&buffer, SessionKind::Domain, 0).unwrap();
    assert_eq!(message.move_handles, Vec::<u32>::new());
    let gsv = Handle::from_raw(message.copy_handles[0]);

    // `Initialize`: one copied handle (the process) and twelve data words.
    let mut words = vec![
        u32::from(TIPC_COMMAND_OFFSET) + command::INITIALIZE,
        0x8000_000C,
        2,
        Handle::CURRENT_PROCESS.raw(),
    ];
    let flags = bootstrap::LOCAL_FLAGS[0];
    let fields = [PROCESS_ID_WORD, flags, 0, 0, 0, 0];
    for field in fields {
        let bytes = field.to_le_bytes();
        words.push(u32::from_le_bytes(bytes[..4].try_into().unwrap()));
        words.push(u32::from_le_bytes(bytes[4..].try_into().unwrap()));
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut message = [0; BUFFER_SIZE];
    message[..bytes.len()].copy_from_slice(&bytes);
    rig.memory.write(TLS, &message).unwrap();
    let regs = rig.call(SvcId::SendSyncRequest, u64::from(gsv.raw()), 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    rig.memory.read(TLS, &mut message).unwrap();
    let reply = Message::parse_reply(&message, SessionKind::Session).unwrap();
    let starquake_ipc::Body::TipcReply { result, data } = reply.body else {
        unreachable!("not a TIPC reply: {:?}", reply.body);
    };
    assert_eq!((result, data.len()), (ResultCode::SUCCESS.raw(), 0));
    // The request reached the root's server, after the requests to `sm:` and `appletOE`.
    assert_eq!(rig.ipc().last(), Some(&(GSV_ROOT, command::INITIALIZE)));

    // Not on a port: only `sm:` is.
    let tables = rig.dispatcher.into_tables();
    let ipc = tables.scheduler.ipc();
    let server = ipc.service(GSV_ROOT).unwrap();
    assert_eq!(ipc.server_name(server), Ok(root::ROOT_NAME));
    assert_eq!(ipc.port(&PortName::new(root::ROOT_NAME).unwrap()), None);
}

/// The graphics server's root is a private entry: `GetService("gsv")` through `sm:` hands out
/// the service manager's own stub (auto-stub policy), and a request on that session is served
/// by `sm:`, never by [`GSV_ROOT`].
#[test]
fn get_service_never_hands_out_the_graphics_server_s_root() {
    let rig = Rig::new();
    let session = rig.service(root::ROOT_NAME);
    let (code, buffer) = rig.send(
        session,
        &libnx(starquake_gsv::protocol::command::INITIALIZE),
    );
    assert_eq!(code, ResultCode::SUCCESS);
    // Auto-stubbed by `sm:`'s server: success, nothing more.
    let (word, data, moved) = reply(&buffer, 0);
    assert_eq!((word, data, moved), (0, Vec::new(), Vec::new()));
    let ipc = rig.ipc();
    assert_eq!(
        ipc.last(),
        Some(&(SM, starquake_gsv::protocol::command::INITIALIZE))
    );
    assert!(
        ipc.iter().all(|(service, _)| *service != GSV_ROOT),
        "{ipc:?}"
    );
}

/// Any applet resource user id the bootstrap names: the root does not check it.
const PROCESS_ID_WORD: u64 = 0x0000_0051_0000_0001;

/// `hid` as libnx starts it (`_hidInitialize`): `CreateAppletResource` hands out a session, and
/// its `GetSharedMemoryHandle` a copied handle to a shared memory.
#[test]
fn hid_hands_out_an_applet_resource_and_its_shared_memory() {
    let rig = Rig::new();
    let hid_session = rig.service(hid::NAME);
    let create = command(&hid::IHidServer, "create_applet_resource");
    let request = libnx(create).send_pid(0).data(&0_u64.to_le_bytes());
    let (code, buffer) = rig.send(hid_session, &request);
    let (word, _, moved) = reply(&buffer, 0);
    assert_eq!((code, word, moved.len()), (ResultCode::SUCCESS, 0, 1));
    let resource = Handle::from_raw(moved[0]);
    let get = command(&hid::IAppletResource::default(), "get_shared_memory_handle");
    let (code, buffer) = rig.send(resource, &libnx(get));
    let message = Message::parse_reply_sized(&buffer, SessionKind::Session, 0).unwrap();
    assert_eq!((code, message.copy_handles.len()), (ResultCode::SUCCESS, 1));
    let ipc = rig.ipc();
    assert_eq!(ipc[ipc.len() - 2..], [(HID, create), (HID, get)]);
}

/// `time:u` as libnx starts it (`_timeInitialize`, then `__libnx_init_time`'s
/// `timeToCalendarTimeWithMyRule`): five interfaces, the shared memory, and a calendar time under
/// UTC.
#[test]
fn time_u_hands_out_its_clocks_its_shared_memory_and_a_calendar_time() {
    let rig = Rig::new();
    let time_session = rig.service(time::NAME);
    let mut zone = None;
    for name in [
        "get_standard_user_system_clock",
        "get_standard_network_system_clock",
        "get_standard_steady_clock",
        "get_time_zone_service",
        "get_standard_local_system_clock",
    ] {
        let request = libnx(command(&time::IStaticService, name));
        let (code, buffer) = rig.send(time_session, &request);
        let (word, _, moved) = reply(&buffer, 0);
        assert_eq!(
            (code, word, moved.len()),
            (ResultCode::SUCCESS, 0, 1),
            "{name}"
        );
        if name == "get_time_zone_service" {
            zone = Some(Handle::from_raw(moved[0]));
        }
    }
    let get = command(&time::IStaticService, "get_shared_memory_native_handle");
    let (code, buffer) = rig.send(time_session, &libnx(get));
    let message = Message::parse_reply_sized(&buffer, SessionKind::Session, 0).unwrap();
    assert_eq!((code, message.copy_handles.len()), (ResultCode::SUCCESS, 1));

    let to_calendar = command(&time::ITimeZoneService, "to_calendar_time_with_my_rule");
    let start = TimeSettings::eden_default().rtc_at_boot;
    let request = libnx(to_calendar).data(&start.to_le_bytes());
    let (code, buffer) = rig.send(zone.unwrap(), &request);
    let (word, data, _) = reply(&buffer, time::calendar::REPLY_SIZE);
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    // 2023-01-01T00:00:00, a Sunday, the first day of the year, in `UTC`.
    assert_eq!(data[..8], [0xE7, 0x07, 1, 1, 0, 0, 0, 0]);
    assert_eq!(data[8..16], [0; 8]);
    assert_eq!(data[16..24], time::calendar::UTC_NAME);
    assert_eq!(rig.ipc().last(), Some(&(TIME_U, to_calendar)));
}
