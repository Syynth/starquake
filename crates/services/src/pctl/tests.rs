//! Parental control driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/pctl.c`: `_pctlCreateService`, `_pctlCmdNoIO`), over a fake
//! kernel. libnx makes the session a domain before it creates the service, so most tests talk
//! to domain objects.

use starquake_ipc::{
    BUFFER_SIZE, Body, CommandHeader, CommandType, Message, ReceiveList, RequestBuilder,
    RequestLayout, SFCO_MAGIC, SessionKind, SpecialHeader,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::UNKNOWN_COMMAND_ID;
use crate::framework::snapshot::encode;
use crate::framework::{Handled, Kernel, Protocol, Session, SnapshotError};

use super::command::factory::{CREATE_SERVICE, CREATE_SERVICE_WITHOUT_INITIALIZE};
use super::command::service::INITIALIZE;

/// `IParentalControlService::IsRestrictionEnabled`, which this port leaves out, so it has no
/// shipping constant (`eden/src/core/hle/service/pctl/parental_control_service.cpp`: `{1031,
/// D<&IParentalControlService::IsRestrictionEnabled>, "IsRestrictionEnabled"}`).
const IS_RESTRICTION_ENABLED: u32 = 1031;

/// The caller's process id: distinct from every other number the tests use.
const PROCESS_ID: u64 = 0x0123_4567_89AB_CDEF;
/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// Both factory commands.
const FACTORY_COMMANDS: [u32; 2] = [CREATE_SERVICE, CREATE_SERVICE_WITHOUT_INITIALIZE];

/// A parental-control server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answered {
    result: ResultCode,
    data: Vec<u8>,
    moves: Vec<u32>,
    objects: Vec<u32>,
}

fn parse(buffer: &[u8], kind: SessionKind) -> Answered {
    let message = Message::parse_reply(buffer, kind).unwrap();
    let Body::CmifReply(reply) = &message.body else {
        unreachable!("not a CMIF reply: {:?}", message.body);
    };
    Answered {
        result: ResultCode::from_raw(reply.header.result),
        data: reply.data.to_vec(),
        moves: message.move_handles.clone(),
        objects: reply.object_ids.clone(),
    }
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// The words of a domain reply with no data and no handles: header, padding, the domain header
/// counting `objects.len()` objects, `SFCO`, version, result, token, then the object ids.
fn domain_reply(result: ResultCode, objects: &[u32]) -> Vec<u8> {
    let mut words = vec![0, 0, 0, 0];
    words.extend([u32::try_from(objects.len()).unwrap(), 0, 0, 0]);
    words.extend([SFCO_MAGIC, 0, result.raw(), 0]);
    words.extend(objects);
    // Raw data runs from after the two header words to the end.
    words[1] = u32::try_from(words.len() - 2).unwrap();
    buffer_of(&words)
}

/// A CMIF reply on a plain session that holds only a result.
fn only(result: ResultCode) -> Vec<u8> {
    buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0])
}

/// A factory request as libnx sends one (`_pctlCreateService`) to object `object`: the process
/// id, and the reserved `u64`.
fn create_in_domain(object: u32, command: u32) -> RequestBuilder {
    RequestBuilder::domain(CommandType::Request, object, command)
        .send_pid(PROCESS_ID)
        .data(&0_u64.to_le_bytes())
}

impl Rig {
    fn of(server: Server) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        let mut server = server;
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn new(name: Name) -> Self {
        Self::of(server(name, UnknownPolicy::Strict))
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// Sends `command` with no payload to object `object` of the rig's domain; the raw reply.
    fn object_raw(&mut self, object: u32, command: u32) -> Vec<u8> {
        let request = RequestBuilder::domain(CommandType::Request, object, command);
        self.send_raw(self.session, &request)
    }

    /// Converts the rig's session to a domain, as libnx does; the root's object id.
    fn convert(&mut self) -> u32 {
        let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
        let reply = parse(&self.send_raw(self.session, &convert), SessionKind::Session);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        u32::from_le_bytes(reply.data.try_into().unwrap())
    }

    /// Converts, sends the factory command `command` as libnx does, and returns the new
    /// service's object id.
    fn service(&mut self, command: u32) -> u32 {
        let root = self.convert();
        let reply = parse(
            &self.send_raw(self.session, &create_in_domain(root, command)),
            SessionKind::Domain,
        );
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply.objects[0]
    }

    /// Whether each [`IParentalControlService`] the server holds is initialized, in the order
    /// its snapshot lists them. A snapshot writes a session as its interface's name (`postcard`:
    /// a one-byte length for a name this short, then its bytes) and then its state's bytes (a
    /// length, then the state), and the state is one `bool`. The length byte tells the
    /// service's name apart from the factory's, which starts with it.
    fn initialized(&self) -> Vec<bool> {
        let saved = self.server.save().unwrap();
        let name = IParentalControlService::interface();
        let mut prefix = vec![u8::try_from(name.len()).unwrap()];
        assert!(prefix[0] < 0x80, "a one-byte length");
        prefix.extend(name.as_bytes());
        prefix.push(1);
        saved
            .windows(prefix.len() + 1)
            .filter(|window| window.starts_with(&prefix))
            .map(|window| match window[prefix.len()] {
                0 => false,
                1 => true,
                other => unreachable!("not a bool: {other}"),
            })
            .collect()
    }
}

// ----- the names and their capabilities -----------------------------------------------------

#[test]
fn the_names_and_capabilities_are_eden_s() {
    // `pctl.cpp`, `LoopProcess`, in its order.
    let names: Vec<&str> = Name::ALL.iter().map(|name| name.as_str()).collect();
    assert_eq!(names, ["pctl", "pctl:a", "pctl:r", "pctl:s"]);
    // `Capability::Application | Capability::SnsPost | Capability::Status |
    // Capability::StereoVision` for `pctl`, `Capability::None` for the others.
    assert_eq!(Name::Pctl.capability().bits(), 0b11_0000_0011);
    for name in [Name::A, Name::S, Name::R] {
        assert_eq!(name.capability(), Capability::NONE, "{}", name.as_str());
    }
}

#[test]
fn the_capability_bits_are_eden_s() {
    // `pctl_types.h`: `None = 0, Application = 1 << 0, SnsPost = 1 << 1, Recovery = 1 << 6,
    // Status = 1 << 8, StereoVision = 1 << 9, System = 1 << 15`.
    let bits = [
        (Capability::NONE, 0),
        (Capability::APPLICATION, 1 << 0),
        (Capability::SNS_POST, 1 << 1),
        (Capability::RECOVERY, 1 << 6),
        (Capability::STATUS, 1 << 8),
        (Capability::STEREO_VISION, 1 << 9),
        (Capability::SYSTEM, 1 << 15),
    ];
    for (capability, expected) in bits {
        assert_eq!(capability.bits(), expected);
    }
    assert_eq!(Capability::KNOWN, 0x8343);
}

/// Every single bit, at both sides of each named one: accepted exactly when Eden names it.
#[test]
fn a_capability_holds_only_the_bits_eden_names() {
    for bit in 0..u32::BITS {
        let bits = 1_u32 << bit;
        let expected = if Capability::KNOWN & bits == 0 {
            Err(UnknownCapability(bits))
        } else {
            Ok(Capability(bits))
        };
        assert_eq!(Capability::from_bits(bits), expected, "bit {bit}");
        assert_eq!(Capability::try_from(bits), expected, "bit {bit}");
    }
    assert_eq!(
        Capability::from_bits(Capability::KNOWN),
        Ok(Capability(Capability::KNOWN))
    );
    // A known bit does not excuse an unknown one beside it.
    let mixed = Capability::APPLICATION.bits() | 1 << 2;
    assert_eq!(Capability::from_bits(mixed), Err(UnknownCapability(mixed)));
    assert_eq!(
        UnknownCapability(mixed).to_string(),
        "capability 0x5 has a bit parental control does not define"
    );
}

#[test]
fn union_and_intersects_are_bitwise() {
    let both = Capability::APPLICATION.union(Capability::SYSTEM);
    assert_eq!(
        both.bits(),
        Capability::APPLICATION.bits() | Capability::SYSTEM.bits()
    );
    assert!(both.intersects(Capability::SYSTEM));
    assert!(both.intersects(Capability::APPLICATION));
    assert!(!both.intersects(Capability::STATUS));
    assert!(!Capability::NONE.intersects(Capability::NONE));
    // Two bits without `Application`'s, and a capability with itself.
    let status_and_system = Capability::STATUS.union(Capability::SYSTEM);
    assert_eq!(
        status_and_system.bits(),
        Capability::STATUS.bits() | Capability::SYSTEM.bits()
    );
    assert_eq!(
        status_and_system.union(status_and_system),
        status_and_system
    );
    assert_eq!(Capability::NONE.union(Capability::NONE), Capability::NONE);
}

/// A capability is decoded through [`Capability::from_bits`], and its layout is a `postcard`
/// `u32`.
#[test]
fn a_capability_is_saved_as_its_word_and_decoded_through_the_check() {
    let pctl = Name::Pctl.capability();
    // 0x303 as a varint: the low seven bits with the high bit set, then the rest.
    assert_eq!(encode(&pctl).unwrap(), [0x83, 0x06]);
    assert_eq!(postcard::from_bytes::<Capability>(&[0x83, 0x06]), Ok(pctl));
    let unknown = encode(&(pctl.bits() | 1 << 2)).unwrap();
    assert_eq!(
        postcard::from_bytes::<Capability>(&unknown),
        Err(postcard::Error::SerdeDeCustom)
    );
}

#[test]
fn a_service_is_saved_as_its_flag() {
    let fresh = IParentalControlService::default();
    assert_eq!(encode(&fresh).unwrap(), [0]);
    let initialized = IParentalControlService { initialized: true };
    assert_eq!(encode(&initialized).unwrap(), [1]);
    assert_eq!(encode(&IParentalControlServiceFactory).unwrap(), [0_u8; 0]);
}

/// The ids, from Eden's tables (`parental_control_service_factory.cpp`: `{0, ..CreateService..}`,
/// `{1, ..CreateServiceWithoutInitialize..}`; `parental_control_service.cpp`: `{1,
/// ..Initialize..}`), and libnx's (`pctl.c`: `_pctlCreateService` with id 0 before 4.0.0 and 1
/// after, then `_pctlCmdNoIO(1)`). Every other test reaches them through the constants, so this
/// is what pins their values.
#[test]
fn the_command_ids_are_eden_s() {
    assert_eq!(
        (
            CREATE_SERVICE,
            CREATE_SERVICE_WITHOUT_INITIALIZE,
            INITIALIZE
        ),
        (0, 1, 1)
    );
}

#[test]
fn no_capability_is_eden_s_result() {
    // `constexpr Result ResultNoCapability{ErrorModule::PCTL, 131}`, `PCTL = 142`.
    assert_eq!(ResultCode::new(PCTL_MODULE, 131), Some(NO_CAPABILITY));
    assert_eq!(NO_CAPABILITY.raw(), 0x1_068E);
    assert_eq!(NO_CAPABILITY.module(), PCTL_MODULE);
    assert_eq!(NO_CAPABILITY.description(), 131);
}

// ----- the factory --------------------------------------------------------------------------

/// Each name opens, and each factory command returns an object in the domain, byte for byte.
#[test]
fn every_name_s_factory_returns_a_service_object() {
    for name in Name::ALL {
        for command in FACTORY_COMMANDS {
            let mut rig = Rig::new(name);
            let root = rig.convert();
            let reply = rig.send_raw(rig.session, &create_in_domain(root, command));
            let service = root + 1;
            assert_eq!(
                reply,
                domain_reply(ResultCode::SUCCESS, &[service]),
                "{} {command}",
                name.as_str()
            );
            assert_eq!(rig.server.interface_count(), 2);
            assert_eq!(rig.kernel.sessions(), []);
            // A fresh service is not initialized (Eden's `States states{}`).
            assert_eq!(rig.initialized(), [false]);
        }
    }
}

/// On a plain session, a factory command moves a handle to a new session, and that session is
/// an `IParentalControlService`: it answers `Initialize`, which the factory does not have.
#[test]
fn a_factory_command_on_a_plain_session_moves_a_session_handle() {
    for command in FACTORY_COMMANDS {
        let mut rig = Rig::new(Name::Pctl);
        let request = RequestBuilder::cmif(CommandType::Request, command)
            .layout(RequestLayout::Libnx)
            .send_pid(PROCESS_ID)
            .data(&0_u64.to_le_bytes());
        let reply = rig.send_raw(rig.session, &request);
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let handle = Handle::from_raw(message.move_handles[0]);
        // Byte for byte: the header with one moved handle, the handle, then `SFCO`, version,
        // result, token, already on a 16-byte boundary. No payload.
        let header = CommandHeader {
            command_type: CommandType::Invalid,
            num_x: 0,
            num_a: 0,
            num_b: 0,
            num_w: 0,
            raw_data_words: 4,
            receive_list: ReceiveList::Disabled,
            has_special_header: true,
        };
        let special = SpecialHeader {
            send_pid: false,
            num_copy_handles: 0,
            num_move_handles: 1,
        };
        let [w0, w1] = header.encode().unwrap();
        let success = ResultCode::SUCCESS.raw();
        let expected = [
            w0,
            w1,
            special.encode().unwrap(),
            handle.raw(),
            SFCO_MAGIC,
            0,
            success,
            0,
        ];
        assert_eq!(reply, buffer_of(&expected), "{command}");
        let service = rig.kernel.resolve_handle(handle).unwrap();
        assert_eq!(rig.kernel.sessions(), [service]);
        let initialize = RequestBuilder::cmif(CommandType::Request, INITIALIZE);
        assert_eq!(
            rig.send_raw(service, &initialize),
            only(ResultCode::SUCCESS)
        );
        assert_eq!(rig.initialized(), [true]);
    }
}

/// Like Eden, which takes the process id from the kernel without checking it was sent and
/// reads nothing from the payload: neither is needed.
#[test]
fn a_factory_request_without_the_process_id_or_payload_still_creates() {
    for command in FACTORY_COMMANDS {
        let mut rig = Rig::new(Name::Pctl);
        let root = rig.convert();
        let reply = rig.object_raw(root, command);
        assert_eq!(
            reply,
            domain_reply(ResultCode::SUCCESS, &[root + 1]),
            "{command}"
        );
    }
}

/// Eden's `CreateService` is its `CreateServiceWithoutInitialize`: the service it returns has
/// not run `Initialize`.
#[test]
fn create_service_does_not_initialize() {
    let mut rig = Rig::new(Name::Pctl);
    rig.service(CREATE_SERVICE);
    assert_eq!(rig.initialized(), [false]);
}

// ----- Initialize ---------------------------------------------------------------------------

/// `Initialize` on each name's service, as a title sends it: success on `pctl`, Eden's
/// `ResultNoCapability` on the three names whose factory has no capability. A failure leaves the
/// service uninitialized; a success marks it, and a second success leaves it marked.
#[test]
fn initialize_answers_by_the_name_s_capability() {
    for name in Name::ALL {
        for command in FACTORY_COMMANDS {
            let mut rig = Rig::new(name);
            let service = rig.service(command);
            let expected = if name == Name::Pctl {
                ResultCode::SUCCESS
            } else {
                NO_CAPABILITY
            };
            let label = format!("{} {command}", name.as_str());
            for _ in 0..2 {
                let reply = rig.object_raw(service, INITIALIZE);
                assert_eq!(reply, domain_reply(expected, &[]), "{label}");
                assert_eq!(rig.initialized(), [name == Name::Pctl], "{label}");
            }
        }
    }
}

/// Eden's check, `capability & (Application | System)`, for each bit alone, and with the two
/// needed bits together.
#[test]
fn initialize_needs_application_or_system() {
    let cases = [
        (Capability::APPLICATION, ResultCode::SUCCESS),
        (Capability::SYSTEM, ResultCode::SUCCESS),
        (
            Capability::APPLICATION.union(Capability::SYSTEM),
            ResultCode::SUCCESS,
        ),
        (Capability::NONE, NO_CAPABILITY),
        (Capability::SNS_POST, NO_CAPABILITY),
        (Capability::RECOVERY, NO_CAPABILITY),
        (Capability::STATUS, NO_CAPABILITY),
        (Capability::STEREO_VISION, NO_CAPABILITY),
        (
            Capability::from_bits(Capability::KNOWN & !INITIALIZE_NEEDS.bits()).unwrap(),
            NO_CAPABILITY,
        ),
    ];
    for (capability, expected) in cases {
        let mut rig = Rig::of(server_with(Name::A, capability, UnknownPolicy::Strict));
        let service = rig.service(CREATE_SERVICE_WITHOUT_INITIALIZE);
        let reply = rig.object_raw(service, INITIALIZE);
        assert_eq!(
            reply,
            domain_reply(expected, &[]),
            "{:#x}",
            capability.bits()
        );
        assert_eq!(rig.initialized(), [expected.is_success()]);
    }
}

/// Each service keeps its own flag: initializing one leaves another as it was.
#[test]
fn each_service_keeps_its_own_flag() {
    let mut rig = Rig::new(Name::Pctl);
    let root = rig.convert();
    let mut services = Vec::new();
    for command in FACTORY_COMMANDS {
        let reply = parse(
            &rig.send_raw(rig.session, &create_in_domain(root, command)),
            SessionKind::Domain,
        );
        services.push(reply.objects[0]);
    }
    rig.object_raw(services[1], INITIALIZE);
    assert_eq!(rig.initialized(), [false, true]);
}

// ----- what is left out ---------------------------------------------------------------------

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new(Name::Pctl);
    let service = rig.service(CREATE_SERVICE_WITHOUT_INITIALIZE);
    let reply = rig.object_raw(service, IS_RESTRICTION_ENABLED);
    assert_eq!(reply, domain_reply(UNKNOWN_COMMAND_ID, &[]));
    assert_eq!(rig.initialized(), [false]);
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
    let factory = IParentalControlServiceFactory.commands();
    assert_eq!(
        factory.name(Protocol::Cmif, CREATE_SERVICE),
        Some("create_service")
    );
    assert_eq!(
        factory.name(Protocol::Cmif, CREATE_SERVICE_WITHOUT_INITIALIZE),
        Some("create_service_without_initialize")
    );
    let service = IParentalControlService::default();
    assert_eq!(
        service.commands().name(Protocol::Cmif, INITIALIZE),
        Some("initialize")
    );
    assert_eq!(service.commands().cmif.len(), 1);
    assert_eq!(
        IParentalControlServiceFactory::interface(),
        "IParentalControlServiceFactory"
    );
    assert_eq!(
        IParentalControlService::interface(),
        "IParentalControlService"
    );
}

// ----- snapshots ----------------------------------------------------------------------------

/// A restored server's service answers as the saved one did, and keeps its flag.
#[test]
fn a_restored_server_s_service_answers_and_keeps_its_flag() {
    let mut rig = Rig::new(Name::Pctl);
    let first = rig.service(CREATE_SERVICE_WITHOUT_INITIALIZE);
    rig.object_raw(first, INITIALIZE);
    let second = parse(
        &rig.send_raw(rig.session, &create_in_domain(1, CREATE_SERVICE)),
        SessionKind::Domain,
    )
    .objects[0];
    let saved = rig.server.save().unwrap();

    let mut restored = server(Name::Pctl, UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    rig.server = restored;
    assert_eq!(rig.initialized(), [true, false]);
    assert_eq!(
        rig.server.family::<Capability>(),
        Some(&Name::Pctl.capability())
    );
    // The restored objects are the saved ones: the second still initializes, and the next
    // object is the one after it.
    assert_eq!(
        rig.object_raw(second, INITIALIZE),
        domain_reply(ResultCode::SUCCESS, &[])
    );
    assert_eq!(rig.initialized(), [true, true]);
    assert_eq!(
        rig.send_raw(rig.session, &create_in_domain(1, CREATE_SERVICE)),
        domain_reply(ResultCode::SUCCESS, &[second + 1])
    );
}

#[test]
fn a_snapshot_is_restored_only_into_its_own_name_s_server() {
    let saved = server(Name::Pctl, UnknownPolicy::Strict).save().unwrap();
    for name in [Name::A, Name::S, Name::R] {
        let mut other = server(name, UnknownPolicy::Strict);
        let before = other.save().unwrap();
        assert_eq!(
            other.restore(&saved),
            Err(SnapshotError::WrongServer {
                expected: name.as_str().to_owned(),
                found: Name::Pctl.as_str().to_owned(),
            })
        );
        assert_eq!(other.save().unwrap(), before);
    }
}

/// A snapshot of a server with the right name and another capability is refused, and the
/// server is left as it was. With the same capability, the same snapshot loads.
#[test]
fn a_snapshot_with_another_capability_is_refused() {
    let mut rig = Rig::of(server_with(
        Name::Pctl,
        Capability::SYSTEM,
        UnknownPolicy::Strict,
    ));
    let service = rig.service(CREATE_SERVICE);
    rig.object_raw(service, INITIALIZE);
    let saved = rig.server.save().unwrap();

    let mut run = server(Name::Pctl, UnknownPolicy::Strict);
    let before = run.save().unwrap();
    assert_eq!(
        run.restore(&saved),
        Err(SnapshotError::FamilyWiring(
            "its parental-control capability is not this server's"
        ))
    );
    assert_eq!(run.save().unwrap(), before);
    assert_eq!(run.family::<Capability>(), Some(&Name::Pctl.capability()));

    let mut same = server_with(Name::Pctl, Capability::SYSTEM, UnknownPolicy::Strict);
    assert_eq!(same.restore(&saved), Ok(()));
}

/// The family state is the snapshot's last bytes (`Option<Vec<u8>>`: present, the length, the
/// state). One whose capability has a bit Eden does not name fails to decode.
#[test]
fn a_snapshot_with_an_unknown_capability_bit_is_refused() {
    let mut saved = server(Name::Pctl, UnknownPolicy::Strict).save().unwrap();
    let family = encode(&Name::Pctl.capability()).unwrap();
    let mut tail = vec![1, u8::try_from(family.len()).unwrap()];
    tail.extend(&family);
    assert!(saved.ends_with(&tail));
    // Bit 2 is between `SnsPost` (1) and `Recovery` (6); it keeps the varint's length.
    let unknown = encode(&(Name::Pctl.capability().bits() | 1 << 2)).unwrap();
    assert_eq!(unknown.len(), family.len());
    let start = saved.len() - family.len();
    saved[start..].copy_from_slice(&unknown);

    let mut run = server(Name::Pctl, UnknownPolicy::Strict);
    let before = run.save().unwrap();
    assert_eq!(
        run.restore(&saved),
        Err(SnapshotError::Decode(postcard::Error::SerdeDeCustom))
    );
    assert_eq!(run.save().unwrap(), before);
}

/// Not checked (module docs, "Snapshots"): a service marked initialized on a server whose
/// capability refuses `Initialize` is restored as saved, and still answers by the server's
/// capability.
#[test]
fn a_service_initialized_where_initialize_fails_is_restored_as_saved() {
    // Initialize on a `pctl:a` server given the Application capability, then write the
    // snapshot's family state as `pctl:a`'s own (`None`, one varint byte, like Application's).
    let mut rig = Rig::of(server_with(
        Name::A,
        Capability::APPLICATION,
        UnknownPolicy::Strict,
    ));
    let service = rig.service(CREATE_SERVICE_WITHOUT_INITIALIZE);
    rig.object_raw(service, INITIALIZE);
    let mut saved = rig.server.save().unwrap();
    let application = encode(&Capability::APPLICATION).unwrap();
    let none = encode(&Capability::NONE).unwrap();
    assert_eq!((application.len(), none.len()), (1, 1));
    assert!(saved.ends_with(&application));
    let last = saved.len() - 1;
    saved[last] = none[0];

    let mut restored = server(Name::A, UnknownPolicy::Strict);
    assert_eq!(restored.restore(&saved), Ok(()));
    rig.server = restored;
    assert_eq!(rig.initialized(), [true]);
    assert_eq!(
        rig.object_raw(service, INITIALIZE),
        domain_reply(NO_CAPABILITY, &[])
    );
    assert_eq!(rig.initialized(), [true]);
}
