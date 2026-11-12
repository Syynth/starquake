//! Unknown commands under both policies: the error under strict, and under auto-stub a success
//! with nothing of the request left in the buffer.
//!
//! What gets logged (once per command under auto-stub) is checked in the crate's
//! `tests/unknown_command_logs.rs`, which explains why it needs a test binary of its own.

use starquake_ipc::{
    BUFFER_SIZE, CMIF_HEADER_BYTES, CommandHeader, RAW_ALIGNMENT_WORDS, RequestBuilder, SFCO_MAGIC,
    TIPC_COMMAND_OFFSET,
};
use starquake_kernel::result::ResultCode;

use super::super::result::UNKNOWN_COMMAND_ID;
use super::super::targets::Domain;
use super::super::{MAX_STUBBED_COMMANDS, Protocol, Session, StubbedCommand, UnknownPolicy};
use super::{Mute, Rig, Toy, buffer_of, domain_request, in_domain, only, plain, request, toy};

/// The CMIF id of the command the toy does not have.
fn missing() -> u32 {
    u32::from(toy::MISSING)
}

const ROOT: u32 = Domain::ROOT_OBJECT_ID;

/// A byte no reply contains, to fill requests with.
const NOISE: u8 = 0xFF;

/// The most argument bytes a CMIF request without descriptors can carry: the buffer less the
/// header words, the padding and the `SFCI` header.
fn cmif_room() -> usize {
    let before_raw = CommandHeader::WORDS.next_multiple_of(RAW_ALIGNMENT_WORDS);
    BUFFER_SIZE - before_raw * size_of::<u32>() - CMIF_HEADER_BYTES
}

/// The most argument bytes a TIPC request can carry: the buffer less the header words.
fn tipc_room() -> usize {
    BUFFER_SIZE - CommandHeader::WORDS * size_of::<u32>()
}

fn stub(interface: &'static str, protocol: Protocol, id: u32) -> StubbedCommand {
    StubbedCommand {
        interface,
        protocol,
        id,
    }
}

#[test]
fn strict_replies_unknown_command_in_each_framing() {
    let mut rig = Rig::strict();
    let failure = UNKNOWN_COMMAND_ID.raw();
    let cmif = rig.send(&request(missing()));
    assert_eq!(cmif, buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, failure, 0]));

    let tipc = rig.send(&RequestBuilder::tipc(toy::MISSING));
    let message_type = u32::from(TIPC_COMMAND_OFFSET + toy::MISSING);
    assert_eq!(tipc, buffer_of(&[message_type, 1, failure]));

    rig.convert();
    let domain = rig.send(&domain_request(ROOT, missing()));
    assert_eq!(
        domain,
        buffer_of(&[0, 10, 0, 0, 0, 0, 0, 0, SFCO_MAGIC, 0, failure, 0])
    );
    assert_eq!(rig.server.stubbed().count(), 0);
}

/// Scoping doc row 96: an auto-stub that left the request's words in the reply crashed a title.
/// The request here fills the buffer with noise; the reply is the result and zeros.
#[test]
fn auto_stub_replies_success_and_zeros_never_the_request() {
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let success = ResultCode::SUCCESS.raw();

    let noisy = request(missing()).data(&vec![NOISE; cmif_room()]);
    assert_eq!(super::built(&noisy).last(), Some(&NOISE));
    let cmif = rig.send(&noisy);
    assert_eq!(cmif, buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, success, 0]));

    let noisy = RequestBuilder::tipc(toy::MISSING).data(&vec![NOISE; tipc_room()]);
    assert_eq!(super::built(&noisy).last(), Some(&NOISE));
    let tipc = rig.send(&noisy);
    let message_type = u32::from(TIPC_COMMAND_OFFSET + toy::MISSING);
    assert_eq!(tipc, buffer_of(&[message_type, 1, success]));

    rig.convert();
    let domain = rig.send(&domain_request(ROOT, missing()).data(&[NOISE; 64]));
    assert_eq!(
        domain,
        buffer_of(&[0, 10, 0, 0, 0, 0, 0, 0, SFCO_MAGIC, 0, success, 0])
    );

    let name = Toy::default().name();
    let stubbed: Vec<_> = rig.server.stubbed().copied().collect();
    assert_eq!(
        stubbed,
        [
            stub(name, Protocol::Cmif, missing()),
            stub(name, Protocol::Tipc, missing())
        ]
    );
}

#[test]
fn an_interface_with_no_commands_answers_by_policy_alone() {
    for (policy, expected) in [
        (UnknownPolicy::Strict, UNKNOWN_COMMAND_ID),
        (UnknownPolicy::AutoStub, ResultCode::SUCCESS),
    ] {
        let mut rig = Rig::new(policy);
        assert_eq!(rig.server.policy(), policy);
        rig.convert();
        let mute = in_domain(&rig.send(&domain_request(ROOT, toy::OPEN_MUTE))).objects[0];
        for id in [0, 1, missing()] {
            let reply = in_domain(&rig.send(&domain_request(mute, id)));
            assert_eq!(reply, only(expected), "{policy:?} command {id}");
        }
        let names: Vec<_> = rig.server.stubbed().map(|c| c.interface).collect();
        let stubbed = if policy == UnknownPolicy::AutoStub {
            3
        } else {
            0
        };
        assert_eq!(names, vec![Mute.name(); stubbed]);
    }
}

/// The record of stubbed commands is bounded: ids are the guest's to choose.
#[test]
fn the_stub_record_stops_growing_exactly_at_its_limit() {
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let limit = u32::try_from(MAX_STUBBED_COMMANDS).unwrap();
    // Ids well above the toy's own: exactly as many as the record holds.
    let first = 0x1000;
    for id in first..first + limit {
        assert_eq!(plain(&rig.send(&request(id))), only(ResultCode::SUCCESS));
    }
    assert_eq!(rig.server.stubbed().count(), MAX_STUBBED_COMMANDS);
    let last = rig.server.stubbed().last().unwrap();
    assert_eq!(
        *last,
        stub(Toy::default().name(), Protocol::Cmif, first + limit - 1)
    );

    // One more is still stubbed, just not recorded.
    assert_eq!(
        plain(&rig.send(&request(first + limit))),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(rig.server.stubbed().count(), MAX_STUBBED_COMMANDS);
    assert!(rig.server.stubbed().all(|c| c.id < first + limit));
}
