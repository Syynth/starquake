//! Root 602 and the camera session, through raw messages laid out by hand: 602 as PR checks it
//! (`packet[0] == 0x26a && packet[1] == 2`), the session's commands as CMIF requests.

use std::sync::Arc;

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_ipc::{BUFFER_SIZE, SFCI_MAGIC, SFCO_MAGIC};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use starquake_services::framework::{Handled, Kernel, Server, UnknownPolicy};
use starquake_trace::ServiceId;

use super::*;
use crate::completion::CompletionQueue;
use crate::protocol::command::OPEN_CAMERA_SESSION;
use crate::root::{RootState, root_server_with_page};
use crate::shared::Shared;
use crate::testing::{STALE, buffer_of, copied, le, result_only, success, tipc_type};

const ROOT: ServiceId = ServiceId(0x0958);
const GSV: ServiceId = ServiceId(0x0959);

struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(ROOT);
        let session = kernel.new_object();
        let queue = Arc::new(CompletionQueue::new());
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV));
        let mut server = root_server_with_page(RootState::new(), UnknownPolicy::Strict, shared)
            .with_service_id(ROOT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// A camera session from 602 with device type `device`.
    fn camera(&mut self, device: u32) -> ObjectId {
        let reply = self.send_on(self.session, &[0x26A, 2, device, 0]);
        let copies = copied(&reply);
        assert_eq!(reply, success(OPEN_CAMERA_SESSION, &copies, &[]));
        self.kernel
            .resolve_handle(Handle::from_raw(copies[0]))
            .unwrap()
    }
}

/// A CMIF request for `command` with `data` words: the header (type 4, raw size: two words of
/// padding to 16 bytes, the four-word CMIF header, the data), the padding, `SFCI`, version 0,
/// the command, token 0, the data.
fn cmif(command: u32, data: &[u32]) -> Vec<u32> {
    let size = 6 + u32::try_from(data.len()).unwrap();
    let mut words = vec![4, size, 0, 0, SFCI_MAGIC, 0, command, 0];
    words.extend(data);
    words
}

/// A CMIF reply: the header (type 0, raw size), padding, `SFCO`, version 0, the result, token
/// 0, then `data`.
fn cmif_reply(result: ResultCode, data: &[u32]) -> Vec<u8> {
    let size = 6 + u32::try_from(data.len()).unwrap();
    let mut words = vec![0, size, 0, 0, SFCO_MAGIC, 0, result.raw(), 0];
    words.extend(data);
    buffer_of(&words)
}

#[test]
fn the_constants_are_the_references_values() {
    assert_eq!(OPEN_DEVICE, 3);
    assert_eq!(COMMAND_21, 21);
    assert_eq!(NO_OUTPUT, [4, 12, 14, 16, 18, 19, 24, 25, 29]);
    assert_eq!(ONE_OUTPUT, [9, 10, 11, 17, 30]);
    assert_eq!(TWO_OUTPUTS, [13]);
    assert_eq!(DEVICE, 0x2_0000_0002);
    assert_eq!(MAX_DEVICE_TYPE, 1);
    assert_eq!(GSV_UNKNOWN_ARGUMENT.raw(), 0x280F3);
    assert_eq!(NAME_BYTES, 16);
    assert_eq!(tipc_type(OPEN_CAMERA_SESSION), 0x26A);
    assert_eq!(tipc_type(601), 0x269);
}

#[test]
fn open_camera_session_takes_device_types_0_and_1() {
    let mut rig = Rig::new();
    let zero = rig.camera(0);
    let one = rig.camera(1);
    assert_ne!(zero, one);
    assert_eq!(rig.kernel.owner(zero), Some(ROOT));
    // 2 is refused; so is a high word that is not 0 (the u64 is 2^32).
    for words in [[0x26A, 2, 2, 0], [0x26A, 2, 0, 1]] {
        assert_eq!(
            rig.send_on(rig.session, &words),
            result_only(OPEN_CAMERA_SESSION, GSV_UNKNOWN_ARGUMENT)
        );
    }
    // Without its high word: refused as short.
    assert_eq!(
        rig.send_on(rig.session, &[0x26A, 1, 0]),
        result_only(OPEN_CAMERA_SESSION, INVALID_CMIF_REQUEST)
    );
}

#[test]
fn open_device_knows_two_names() {
    let mut rig = Rig::new();
    let camera = rig.camera(0);
    let name = |text: &[u8]| -> Vec<u32> {
        let mut bytes = [0_u8; NAME_BYTES];
        bytes[..text.len()].copy_from_slice(text);
        bytes
            .chunks(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    };
    let device = [0x0000_0002, 0x0000_0002];
    for known in [&b"Default"[..], b"External"] {
        assert_eq!(
            rig.send_on(camera, &cmif(OPEN_DEVICE, &name(known))),
            cmif_reply(ResultCode::SUCCESS, &device),
            "{known:?}"
        );
    }
    // A name compares up to its first zero: bytes after it do not count.
    let mut padded = name(b"Default");
    padded[3] = 0x4142_4300;
    assert_eq!(
        rig.send_on(camera, &cmif(OPEN_DEVICE, &padded)),
        cmif_reply(ResultCode::SUCCESS, &device)
    );
    for unknown in [&b"default"[..], b"Defaul", b"DefaultX", b"", b"External0123456"] {
        assert_eq!(
            rig.send_on(camera, &cmif(OPEN_DEVICE, &name(unknown))),
            cmif_reply(GSV_UNKNOWN_ARGUMENT, &[]),
            "{unknown:?}"
        );
    }
    // Sixteen bytes with no zero: the whole name, not a known one.
    let full = [0x4141_4141; 4];
    assert_eq!(
        rig.send_on(camera, &cmif(OPEN_DEVICE, &full)),
        cmif_reply(GSV_UNKNOWN_ARGUMENT, &[])
    );
}

#[test]
fn every_probe_command_answers_pr_s_shape() {
    let mut rig = Rig::new();
    let camera = rig.camera(1);
    for command in NO_OUTPUT {
        assert_eq!(
            rig.send_on(camera, &cmif(command, &[7, 8, 9])),
            cmif_reply(ResultCode::SUCCESS, &[]),
            "{command}"
        );
    }
    for command in ONE_OUTPUT {
        assert_eq!(
            rig.send_on(camera, &cmif(command, &[])),
            cmif_reply(ResultCode::SUCCESS, &[0, 0]),
            "{command}"
        );
    }
    for command in TWO_OUTPUTS {
        assert_eq!(
            rig.send_on(camera, &cmif(command, &[1])),
            cmif_reply(ResultCode::SUCCESS, &[0, 0, 0, 0]),
            "{command}"
        );
    }
    assert_eq!(
        rig.send_on(camera, &cmif(COMMAND_21, &[1, 0])),
        cmif_reply(ResultCode::SUCCESS, &[])
    );
    // A command PR does not list goes to the policy.
    for command in [0, 1, 2, 5, 15, 20, 22, 31] {
        assert_eq!(
            rig.send_on(camera, &cmif(command, &[])),
            cmif_reply(UNKNOWN_COMMAND_ID, &[]),
            "{command}"
        );
    }
}

#[test]
fn the_camera_session_keeps_nothing() {
    assert_eq!(postcard::to_allocvec(&CameraSession).unwrap(), Vec::<u8>::new());
    assert!(is_device_name(b"Default\0\0\0\0\0\0\0\0\0"));
    assert!(!is_device_name(b"Default\x01\0\0\0\0\0\0\0\0"));
}
