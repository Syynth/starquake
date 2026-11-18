//! The root's 902, 906 and 1002 (module docs, "Video (slice 15)"): sessions on the video server
//! in a root wired to one, the stub's placeholder in one that is not, and the server's number
//! checked on a snapshot. Header words are PR's (`OpenVideoDecoderContext`: `packet[0] ==
//! 0x396 && packet[1] == 0` for 902; `DecodeVideoContext`: `0x39a, 3` for 906;
//! `OpenVideoEncoderContext`: `0x3fa, 0` for 1002).

use super::*;

/// The number the tests' video server is known by.
const VIDEO: ServiceId = ServiceId(0x0902);

/// 902, as PR checks it.
const OPEN_902_WORDS: [u32; 2] = [0x396, 0];
/// 1002, as PR checks it.
const OPEN_1002_WORDS: [u32; 2] = [0x3FA, 0];

/// 906 with a remote-memory byte and a library version.
fn open_906(remote: u32, version: u64) -> Vec<u32> {
    let mut words = vec![0x39A, 3, remote];
    words.extend(halves(version));
    words
}

#[test]
fn the_video_header_words_are_the_command_ids() {
    assert_eq!(OPEN_902_WORDS[0], tipc_type(OPEN_VIDEO_DECODER_CONTEXT_902));
    assert_eq!(open_906(0, 0)[0], tipc_type(OPEN_VIDEO_DECODER_CONTEXT));
    assert_eq!(OPEN_1002_WORDS[0], tipc_type(OPEN_VIDEO_ENCODER_CONTEXT));
    assert_eq!(VIDEO_LIBRARY_VERSION, 0x3_0003_0000);
}

/// Each of 902, 906 and 1002 hands out a session on the video server in a wired root, before
/// the bootstrap and after it, with 906's values PR refuses (a remote-memory byte, another
/// library version) as well as its own; an unwired root answers each with the placeholder.
#[test]
fn each_video_open_hands_out_a_session_on_the_video_server() {
    let requests = [
        OPEN_902_WORDS.to_vec(),
        open_906(0, VIDEO_LIBRARY_VERSION),
        open_906(1, VIDEO_LIBRARY_VERSION),
        open_906(0x102, 7),
        OPEN_1002_WORDS.to_vec(),
    ];
    for (state, server) in [
        (RootState::new(), None),
        (RootState::new().with_gpu_server(GPU), None),
        (RootState::new().with_video_server(VIDEO), Some(VIDEO)),
    ] {
        let mut rig = wired_with(state);
        for words in &requests {
            assert_eq!(opened(&mut rig, words), server, "{state:?} {words:x?}");
        }
        rig.initialize(&local());
        rig.bind_process();
        rig.complete_bootstrap();
        for words in &requests {
            assert_eq!(opened(&mut rig, words), server, "{state:?} {words:x?}");
        }
        assert_eq!(rig.state().video_server(), server);
        // No bootstrap command was out of order: the opens moved nothing of the phase.
        assert_eq!(rig.state().out_of_order(), 0);
    }
}

/// A wired 906 reads its version: one without it is refused, connecting nothing; an unwired
/// one is the stub, which reads nothing.
#[test]
fn a_wired_906_reads_its_words() {
    let mut rig = wired_with(RootState::new().with_video_server(VIDEO));
    let short = [0x39A, 2, 0, 0];
    assert_eq!(
        rig.send(&short),
        result_only(OPEN_VIDEO_DECODER_CONTEXT, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.kernel.take_connections().len(), 0);
    let mut stub = wired_with(RootState::new());
    assert_eq!(opened(&mut stub, &short), None);
}

/// A wired 902, 906 or 1002 replies a success word and one copied handle, nothing more.
#[test]
fn a_video_open_reply_is_one_copied_handle_and_success() {
    for (words, command) in [
        (OPEN_902_WORDS.to_vec(), OPEN_VIDEO_DECODER_CONTEXT_902),
        (
            open_906(0, VIDEO_LIBRARY_VERSION),
            OPEN_VIDEO_DECODER_CONTEXT,
        ),
        (OPEN_1002_WORDS.to_vec(), OPEN_VIDEO_ENCODER_CONTEXT),
    ] {
        let mut rig = wired_with(RootState::new().with_video_server(VIDEO));
        let reply = rig.send(&words);
        let copies = copied(&reply);
        assert_eq!(copies.len(), 1);
        assert_eq!(reply, success(command, &copies, &[]));
    }
}

/// A snapshot whose root names another video server, or none, is refused, and changes nothing;
/// with the gpu server wrong too, the gpu server's refusal comes first.
#[test]
fn a_snapshot_with_another_video_server_is_refused() {
    let run = RootState::new()
        .with_gpu_server(GPU)
        .with_video_server(VIDEO);
    let other = ServiceId(0x0999);
    for saved in [
        RootState::new().with_gpu_server(GPU),
        RootState::new()
            .with_gpu_server(GPU)
            .with_video_server(other),
    ] {
        let bytes = wired_with(saved).server.save().unwrap();
        let mut rig = wired_with(run);
        assert_eq!(
            rig.server.restore_beside(&bytes, &rig.kernel),
            Err(SnapshotError::FamilyWiring(
                "its root names another video server"
            )),
            "{saved:?}"
        );
        assert_eq!(rig.state(), run);
    }
    let both = RootState::new()
        .with_gpu_server(other)
        .with_video_server(other);
    let bytes = wired_with(both).server.save().unwrap();
    let mut rig = wired_with(run);
    assert_eq!(
        rig.server.restore_beside(&bytes, &rig.kernel),
        Err(SnapshotError::FamilyWiring(
            "its root names another gpu server"
        ))
    );
}
