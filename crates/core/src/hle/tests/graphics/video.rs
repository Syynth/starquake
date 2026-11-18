//! The video sessions as a run installs the graphics server ([`install_with_graphics`]), behind
//! the real dispatcher: the root's 902, 906 and 1002 hand out sessions on the video server, a
//! decoder service's sync handle counts the frames it finishes, and the guest reads that count
//! through the syncpt session (graphics-server port slice 15).
//!
//! Requests are laid out by hand from the headers PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/native_video_decoder.cpp` and
//! `graphics_supervisor.cpp` @ e5fb3b97: `CreateDecoder` `0x11, 0`, `RegisterImage` `0x147,
//! 4`, `Submit` `0x10007d, 3`, `GetOutputProperties` `0xe5, 0`, `CalculateVideoImageRequirements`
//! `0x3ef, 17`; NX2's 906 as row 146 gives it: a `u8`, then the library version).

use starquake_gsv::native::{BIND_PROCESS_MEMORY, BindRequest};
use starquake_gsv::video::command as video_command;
use starquake_gsv::video::counter_id;
use starquake_gsv::video::layout::{CODE_NV12, CODE_NV12_NX2, PREFERENCE_WORDS, WORD_CODE, WORD_EXTENT};

use super::*;

/// Where the pool's native memory says it is bound: the tables are filled directly, with a test
/// token, so no page of the rig is locked.
const POOL_AT: u64 = 0x9000_0000;

/// A pool of `count` 2x2 NV12 images (PR's layout, 4096 bytes each) put straight into the run's
/// native tables: its first image id.
fn pool(graphics: &Graphics, count: u32) -> u32 {
    let mut locked = graphics.shared().lock();
    let native = locked.native();
    let size = u64::from(count) * PAGE_SIZE;
    let memory = native.create_memory(0, size).unwrap();
    let request = BindRequest {
        id: memory,
        address: POOL_AT,
        size,
        flags: BIND_PROCESS_MEMORY,
        offset: 0,
    };
    let admission = native.check_bind(&request).unwrap();
    admission.bind(device_share_token(ProcessId::from_parts(1, 1), POOL_AT, size));
    let mut preference = [0; PREFERENCE_WORDS];
    preference[WORD_CODE] = CODE_NV12;
    preference[WORD_EXTENT] = 1 | (1 << 16);
    native.create_pool(&preference, count, memory).unwrap()
}

/// The decoder service a context's 1 makes.
fn decoder(rig: &Rig, context: Handle) -> Handle {
    let made = rig.tipc(context, &[tipc_type(video_command::CREATE_DECODER), 0]);
    assert_eq!((made.result, made.copied.len()), (0, 1));
    made.copied[0]
}

/// 109 with one A descriptor of 16 bytes at the rig's name page, the timestamp, and no end of
/// stream: PR's `0x10007d, 3` (the A descriptor count is bits 20 to 23 of word 0).
fn submit(timestamp: u64) -> Vec<u32> {
    let address = u32::try_from(NAME).unwrap();
    let mut words = vec![tipc_type(video_command::SUBMIT) | (1 << 20), 3, 0x10, address, 0];
    words.extend(halves(timestamp));
    words.push(0);
    words
}

/// The run's root hands out the video server's sessions with 902, 906 and 1002, each recorded
/// under the video server's number; a decoder's sync handle names a syncpoint whose counter the
/// guest reads through the syncpt session, and it reaches the handle's value when the frame is
/// submitted, not before. 991 is answered from the native tables (NX2's row 96 size for
/// 640x360 NV12 under code 4), not by the stub's zeros.
#[test]
fn the_root_hands_out_video_sessions_that_count_frames() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let root = rig.graphics_root();
    let syncpt = rig.family(root, root_command::GET_SYNCPOINT_SESSION);

    let mut words = vec![tipc_type(root_command::CALCULATE_VIDEO_IMAGE_REQUIREMENTS), 17];
    let mut preference = [0; PREFERENCE_WORDS];
    preference[WORD_CODE] = CODE_NV12_NX2;
    preference[WORD_EXTENT] = (640 - 1) | ((360 - 1) << 16);
    words.extend(preference);
    words.push(1);
    let sized = rig.tipc(root, &words);
    let mut expected = halves(0x64000).to_vec();
    expected.extend(halves(0x2000));
    assert_eq!((sized.result, sized.data), (0, expected));

    let first = pool(&graphics, 3);
    let context = rig.family(root, root_command::OPEN_VIDEO_DECODER_CONTEXT_902);
    let service = decoder(&rig, context);
    let register = [tipc_type(video_command::REGISTER_IMAGE), 4, first, 0, 0, 0];
    let handle = rig.tipc(service, &register);
    assert_eq!(handle.result, 0);
    let counter = handle.data[0];
    let value = handle.data[2];
    assert_eq!(handle.data[1], 0);
    assert_eq!(handle.data[3..], [0; 5]);
    // The counter is a syncpoint's `SyncCounterId`: syncpt 103 (`0x77, 1`) reads it. Before the
    // frame it is one short of the handle's value.
    let syncpoint = starquake_gsv::syncpt::syncpoint_of(counter).unwrap();
    assert_eq!(counter, counter_id(syncpoint));
    let query = [tipc_type(syncpt_command::QUERY_COUNTER), 1, counter];
    assert_eq!(rig.tipc(syncpt, &query).data, [value - 1, 0, 0]);
    let submitted = rig.tipc(service, &submit(0x55));
    assert_eq!((submitted.result, submitted.data), (0, vec![0x10, 0]));
    assert_eq!(rig.tipc(syncpt, &query).data, [value, 0, 0]);
    let output = rig.tipc(service, &[tipc_type(video_command::GET_OUTPUT_PROPERTIES), 0]);
    assert_eq!(output.result, 0);
    assert_eq!(output.data[0], first);
    assert_eq!(output.data[6..8], halves(0x55 * 1000));

    // 906 (a u8, the library version 0x3_0003_0000) and 1002 hand out contexts on the same
    // server whose decoders work: each registers an image and counts on its own syncpoint.
    let mut version = vec![tipc_type(root_command::OPEN_VIDEO_DECODER_CONTEXT), 3, 0];
    version.extend(halves(0x3_0003_0000));
    let newer = rig.tipc(root, &version);
    assert_eq!((newer.result, newer.copied.len()), (0, 1));
    let encoder = rig.family(root, root_command::OPEN_VIDEO_ENCODER_CONTEXT);
    for (index, context) in [newer.copied[0], encoder].into_iter().enumerate() {
        let service = decoder(&rig, context);
        let image = first + 1 + u32::try_from(index).unwrap();
        let register = [tipc_type(video_command::REGISTER_IMAGE), 4, image, 0, 0, 0];
        let handle = rig.tipc(service, &register);
        assert_eq!(handle.result, 0);
        assert_ne!(handle.data[0], counter);
        let submitted = rig.tipc(service, &submit(1));
        assert_eq!(submitted.result, 0);
        let query = [tipc_type(syncpt_command::QUERY_COUNTER), 1, handle.data[0]];
        assert_eq!(rig.tipc(syncpt, &query).data, [handle.data[2], 0, 0]);
    }

    let ipc = rig.ipc();
    let opens: Vec<_> = ipc
        .iter()
        .filter(|(_, command)| {
            [
                root_command::OPEN_VIDEO_DECODER_CONTEXT_902,
                root_command::OPEN_VIDEO_DECODER_CONTEXT,
                root_command::OPEN_VIDEO_ENCODER_CONTEXT,
            ]
            .contains(command)
        })
        .copied()
        .collect();
    assert_eq!(
        opens,
        [
            (GSV_ROOT, root_command::OPEN_VIDEO_DECODER_CONTEXT_902),
            (GSV_ROOT, root_command::OPEN_VIDEO_DECODER_CONTEXT),
            (GSV_ROOT, root_command::OPEN_VIDEO_ENCODER_CONTEXT),
        ]
    );
    let video: Vec<u32> = ipc
        .iter()
        .filter(|(service, _)| *service == GSV_VIDEO)
        .map(|(_, command)| *command)
        .collect();
    let (create, register, submit, output) = (
        video_command::CREATE_DECODER,
        video_command::REGISTER_IMAGE,
        video_command::SUBMIT,
        video_command::GET_OUTPUT_PROPERTIES,
    );
    assert_eq!(
        video,
        [
            create, register, submit, output, create, register, submit, create, register, submit
        ]
    );
}
