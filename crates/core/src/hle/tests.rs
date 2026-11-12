//! The run's services behind the real dispatcher, on every host: the conversation the homebrew
//! fixture has with them, request by request, as libnx lays its requests out.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use starquake_cpu::GuestRegs;
use starquake_ipc::{
    BUFFER_SIZE, BufferDescriptor, CommandType, Message, ReceiveDescriptor, RequestBuilder,
    RequestLayout, SessionKind, StaticDescriptor,
};
use starquake_kernel::dispatch::{Caller, Outcome};
use starquake_kernel::handle::Handle;
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
use starquake_services::framework::{
    DEFAULT_POINTER_BUFFER_SIZE, NOT_SUPPORTED, Protocol, Session,
};
use starquake_services::fsp::{FspSrv, IFile, IFileSystem};
use starquake_services::set::{self, FIRMWARE_VERSION_SIZE, FirmwareVersion, SetSys, layout};
use starquake_services::sm::{MAX_NAME_LEN, ServiceName, Sm};
use starquake_trace::{EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId, VecSink};

use super::*;

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
        install(&dispatcher, services).unwrap();
        let recorder = Recorder::new();
        let events = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&events));
        Self {
            dispatcher,
            memory: FlatMemory::new(BASE, usize::try_from(PAGES * PAGE_SIZE).unwrap()),
            env,
            recorder,
            events,
        }
    }

    fn call(&self, id: SvcId, x0: u64, x1: u64) -> GuestRegs {
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
        assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
        regs
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
    assert_eq!(service_name(ServiceId(0)), None);
    assert_eq!(service_name(ServiceId(4)), None);
}

#[test]
fn the_service_manager_is_on_its_port_and_set_sys_on_none() {
    let dispatcher = dispatcher();
    install(&dispatcher, no_host()).unwrap();
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
    install(&dispatcher, no_host()).unwrap();
    let port = PortName::new(sm::PORT_NAME).unwrap();
    assert_eq!(
        install(&dispatcher, no_host()),
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
    // Every session and handle is gone.
    let tables = rig.dispatcher.into_tables();
    assert!(tables.handles.is_empty());
    assert!(tables.scheduler.objects().is_empty());
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

    // PROVISIONAL (module docs): a run gives `fsp-srv` no executor, so a read is refused.
    let buffer_room = BufferDescriptor {
        address: FIRMWARE,
        size: 4,
        flags: 0,
    };
    let payload = [
        &0_u32.to_le_bytes()[..],
        &[0; 4],
        &0_i64.to_le_bytes(),
        &4_i64.to_le_bytes(),
    ]
    .concat();
    let read = domain(file, &IFile::default(), "read")
        .data(&payload)
        .receive_buffer(buffer_room);
    let (code, buffer) = rig.send(fsp, &read);
    assert_eq!(code, ResultCode::SUCCESS);
    assert_eq!(domain_reply(&buffer, 0).0, NOT_SUPPORTED.raw());

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
