//! Snapshots: a server is saved, restored and carries on as the saved run would have
//! (`docs/specs/observability.md`, "Snapshots").

use std::sync::Arc;

use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use serde::{Deserialize, Serialize};
use starquake_ipc::{BUFFER_SIZE, RequestBuilder};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::super::defer::{Framing, Pending, TokenId};
use super::super::targets::Domain;
use super::super::{
    Admission, DEFAULT_WORKERS, Interfaces, Reply, ReplyToken, RequestQueue, SNAPSHOT_MAGIC,
    SNAPSHOT_VERSION, Server, SnapshotError, UnknownPolicy,
};
use super::{
    Mute, Rig, Tally, Toy, Wakes, child, domain_request, parked, parked_one, request, toy,
    toy_interfaces, toy_server,
};

/// The object id of a domain's first added object.
const FIRST: u32 = Domain::ROOT_OBJECT_ID + 1;

/// A rig in the middle of a run, with one of everything a snapshot holds, and what a test needs
/// to carry on from there.
struct Scene {
    rig: Rig,
    /// A session whose toy has a deferred CMIF request, and one sender waiting behind it.
    plain: ObjectId,
    /// A session on a child interface.
    child: ObjectId,
    /// A domain with the toy as its root, a closed id ([`FIRST`]) and a child after it, and a
    /// deferred request.
    domain: ObjectId,
    /// A session with a deferred TIPC request.
    tipc: ObjectId,
    /// The tokens of the three deferred requests, in the order of the fields above.
    tokens: [TokenId; 3],
}

fn scene() -> Scene {
    let mut rig = Rig::strict();
    let plain = rig.session;
    let child = rig.open_child(0xC1);
    let domain = rig.second_session();
    rig.send_on(
        domain,
        &super::control_request(super::CONVERT_CURRENT_OBJECT_TO_DOMAIN),
    );
    for tag in [0xD1_u32, 0xD2] {
        let open = domain_request(Domain::ROOT_OBJECT_ID, toy::OPEN_CHILD);
        rig.send_on(domain, &open.data(&tag.to_le_bytes()));
    }
    rig.send_on(domain, &RequestBuilder::domain_close(FIRST));
    let tipc = rig.second_session();
    // State in a session (the toy remembers its events) and in the family.
    rig.send(&request(toy::MAKE_EVENT));
    rig.send(&request(toy::BUMP));
    rig.send(&request(toy::BUMP));

    let wait = domain_request(Domain::ROOT_OBJECT_ID, toy::WAIT);
    let tokens = [
        rig.defer_on(plain, &request(toy::WAIT).data(&[1; 4])),
        rig.defer_on(domain, &wait),
        rig.defer_on(tipc, &RequestBuilder::tipc(toy::TIPC_WAIT)),
    ];
    assert_eq!(rig.server.submit(plain), Ok(Admission::Queued));
    Scene {
        rig,
        plain,
        child,
        domain,
        tipc,
        tokens,
    }
}

/// Carries a run on from [`scene`] and returns every reply, in order. `tokens` are live tokens
/// for the scene's three deferred requests, in the order they were taken.
fn carry_on(scene: &mut Scene, tokens: Vec<ReplyToken>) -> Vec<Vec<u8>> {
    let (plain, child, domain, tipc) = (scene.plain, scene.child, scene.domain, scene.tipc);
    let rig = &mut scene.rig;
    // A session's own state.
    let mut replies = vec![
        rig.send_on(child, &request(child::TAG)),
        rig.send_on(child, &request(child::SET_TAG).data(&7_u32.to_le_bytes())),
        rig.send_on(child, &request(child::TAG)),
    ];
    // The deferred requests are answered, the last one first, each in its own framing.
    for (token, value) in tokens.into_iter().zip([0xA1_u32, 0xA2, 0xA3]).rev() {
        let mut reply = Reply::default();
        reply.payload().write_u32(value);
        token.complete(reply);
    }
    let mut answered = Vec::new();
    for _ in 0..3 {
        let (session, buffer) = rig.deliver();
        answered.push(session);
        replies.push(buffer);
    }
    assert_eq!(answered, [tipc, domain, plain]);
    // The sender that waited behind the deferred request has its turn, and finds the family's
    // state and its session's.
    assert_eq!(rig.server.start_next(), Some(plain));
    replies.push(rig.send_on(plain, &request(toy::BUMP)));
    assert_eq!(rig.server.start_next(), None);
    // The domain hands out the id that was closed last, then a new one.
    for tag in [0xE1_u32, 0xE2] {
        let open = domain_request(Domain::ROOT_OBJECT_ID, toy::OPEN_CHILD);
        replies.push(rig.send_on(domain, &open.data(&tag.to_le_bytes())));
    }
    replies.push(rig.send_on(domain, &domain_request(FIRST, child::TAG)));
    replies.push(rig.send_on(domain, &domain_request(FIRST + 1, child::TAG)));
    replies.push(rig.send_on(tipc, &request(toy::BUMP)));
    // The next deferred request gets the token id after the last one the saved run handed out.
    let token = rig.defer_on(tipc, &request(toy::WAIT));
    replies.push(token.raw().to_le_bytes().to_vec());
    drop(parked());
    replies.push(rig.deliver().1);
    replies
}

#[test]
fn a_restored_server_carries_on_as_the_saved_one_would_have() {
    let mut scene = scene();
    let wakes = Arc::clone(&scene.rig.wakes);
    let saved = scene.rig.server.save().unwrap();
    let handles = scene.rig.kernel.handle_count();

    // The run that was saved carries on.
    let original = carry_on(&mut scene, parked());
    let next_token = scene.tokens[2].raw() + 1;
    assert!(original.contains(&next_token.to_le_bytes().to_vec()));
    // Nothing in the continuation gave out a handle, so the kernel is as it was.
    assert_eq!(scene.rig.kernel.handle_count(), handles);
    assert_ne!(scene.rig.server.save().unwrap(), saved);

    // The same server goes back to the snapshot and carries on again.
    scene.rig.server.restore(&saved).unwrap();
    assert_eq!(scene.rig.server.save().unwrap(), saved);
    let tokens = scene.tokens.map(|id| scene.rig.server.reissue(id).unwrap());
    assert_eq!(carry_on(&mut scene, tokens.into()), original);

    // A server that never saw the run, made the way the first was, does the same.
    scene.rig.server = toy_server(UnknownPolicy::Strict, &wakes);
    scene.rig.server.restore(&saved).unwrap();
    assert_eq!(
        (
            scene.rig.server.session_count(),
            scene.rig.server.interface_count()
        ),
        (4, 5)
    );
    assert_eq!(
        scene.rig.server.family::<Tally>(),
        Some(&Tally { bumps: 2 })
    );
    assert_eq!(
        scene.rig.server.pending().collect::<Vec<_>>(),
        [
            (scene.tokens[0], scene.plain),
            (scene.tokens[1], scene.domain),
            (scene.tokens[2], scene.tipc),
        ]
    );
    let tokens = scene.tokens.map(|id| scene.rig.server.reissue(id).unwrap());
    assert_eq!(carry_on(&mut scene, tokens.into()), original);
}

#[test]
fn tokens_from_before_a_restore_lead_nowhere() {
    let mut scene = scene();
    let saved = scene.rig.server.save().unwrap();
    let stale = parked();
    let woken = scene.rig.wakes.count();
    scene.rig.server.restore(&saved).unwrap();
    for token in stale {
        token.fail(ResultCode::SUCCESS);
    }
    // The restored server was not told and has nothing to deliver; its requests still wait.
    assert_eq!(scene.rig.wakes.count(), woken);
    assert!(scene.rig.server.take_completed().is_none());
    assert_eq!(scene.rig.server.pending().count(), 3);
    // A token for a request that is not pending is not handed out.
    let unknown = TokenId(scene.tokens[2].raw() + 1);
    assert!(scene.rig.server.reissue(unknown).is_none());
}

#[test]
fn the_byte_layout_is_pinned() {
    let scene = scene();
    let bytes = scene.rig.server.save().unwrap();
    assert_eq!(bytes, GOLDEN);
    // The constants and the commands in the bytes are the shipping ones.
    assert_eq!(bytes[..4], SNAPSHOT_MAGIC);
    assert_eq!(u32::from(bytes[4]), SNAPSHOT_VERSION);
    assert!(GOLDEN.contains(&u8::try_from(toy::WAIT).unwrap()));
    assert!(GOLDEN.contains(&u8::try_from(toy::TIPC_WAIT).unwrap()));
    drop(parked());
}

/// The whole snapshot of [`scene`], byte for byte (`postcard`: integers are varints, a
/// sequence or a string starts with its length, an enum with its variant's index, an `Option`
/// with 0 or 1). Ids are an index and then a generation.
///
/// The scene's kernel objects, by index: 0 the first session, 1 the child's session, 2 the
/// domain's session, 3 the TIPC session, 4 the event the first toy made. Its targets, by slot:
/// 0 the first toy, 1 its child, 2 the domain's root toy, 3 the domain, 4 the TIPC session's toy
/// (in the slot the closed domain object left, so generation 2), 5 the domain's second child.
#[rustfmt::skip]
const GOLDEN: &[u8] = &[
    // Header: magic, version.
    b'S', b'Q', b'S', b'V', 1,
    // The server's name.
    3, b't', b'o', b'y',
    // Sessions: four, each a session object and the target it leads to.
    4,
    0, 1,  0, 1,
    1, 1,  1, 1,
    2, 1,  3, 1,
    3, 1,  4, 2,
    // Targets: six slots, each occupied (1) with its generation, then the row: its reference
    // count, the kind of target (0 an interface, 1 a domain), and the target.
    6,
    1, 1,  1, 0, 3, b'T', b'o', b'y', 3, 1, 4, 1, // state: one event, object 4
    1, 1,  1, 0, 5, b'C', b'h', b'i', b'l', b'd', 2, 0xC1, 0x01, // state: tag 0xC1
    1, 1,  1, 0, 3, b'T', b'o', b'y', 1, 0, // state: no events
    // The domain: three entries (the root, a closed id, the second child), and the free list
    // naming entry 1.
    1, 1,  1, 1, 3,  1, 2, 1,  0,  1, 5, 1,  1, 1,
    1, 2,  1, 0, 3, b'T', b'o', b'y', 1, 0,
    1, 1,  1, 0, 5, b'C', b'h', b'i', b'l', b'd', 2, 0xD2, 0x01, // state: tag 0xD2
    // The slab's free list (empty), its slot limit (`u32::MAX`) and its count of rows.
    0,  0xFF, 0xFF, 0xFF, 0xFF, 0x0F,  6,
    // The queue: one worker, nothing running, nobody ready; three sessions, each with a
    // deferred request (Some, variant 1) and its count of waiting senders.
    1, 0, 0,
    3,
    0, 1,  1, 1,  1,
    2, 1,  1, 1,  0,
    3, 1,  1, 1,  0,
    // Deferred requests: three, each a token id, its session, its framing (0 CMIF, 1 domain,
    // 2 TIPC with the command id), the command and the size of its arguments.
    3,
    0,  0, 1,  0,     14, 4,
    1,  2, 1,  1,     14, 0,
    2,  3, 1,  2, 3,  3,  0,
    // The id of the next token.
    3,
    // The family's state: present, one byte, two bumps.
    1, 1, 2,
];

/// Gives `bytes` to a fresh toy server as a snapshot, and then uses the server. Nothing here
/// may panic, whatever the bytes are: a snapshot is a file.
fn restore_and_use(bytes: &[u8]) -> Result<(), TestCaseError> {
    let wakes = Arc::new(Wakes::default());
    let mut server = toy_server(UnknownPolicy::Strict, &wakes);
    let before = server.save().unwrap();
    if server.restore(bytes).is_err() {
        // Refused: the server is as it was.
        prop_assert_eq!(server.save().unwrap(), before);
        return Ok(());
    }
    // Accepted: the state passed every check, so it can be saved again, read back, and used.
    let saved = server.save().unwrap();
    prop_assert_eq!(server.restore(&saved), Ok(()));
    let mut kernel = super::FakeKernel::new(0).unwrap();
    let sessions: Vec<ObjectId> = server.sessions.keys().copied().collect();
    for session in sessions {
        let _ = server.submit(session);
        let mut buffer = super::built(&request(toy::OPEN_CHILD).data(&[1, 0, 0, 0]));
        let _ = server.handle(&mut kernel, session, &mut buffer);
        let mut buffer = super::built(&RequestBuilder::close());
        let _ = server.handle(&mut kernel, session, &mut buffer);
    }
    let tokens: Vec<TokenId> = server.pending().map(|(token, _)| token).collect();
    for token in tokens {
        if let Some(token) = server.reissue(token) {
            token.fail(ResultCode::SUCCESS);
        }
    }
    while let Some(completed) = server.take_completed() {
        let mut buffer = vec![0; BUFFER_SIZE];
        let _ = server.deliver(&mut kernel, completed, &mut buffer);
    }
    while server.start_next().is_some() {}
    Ok(())
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 1024,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x5351_534E_4150_5331),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// A snapshot with one byte changed, or cut short, is refused or is a consistent state;
    /// either way nothing panics and nothing is half restored.
    #[test]
    fn a_damaged_snapshot_is_refused_or_usable(at: Index, byte: u8, cut: Index) {
        let mut damaged = GOLDEN.to_vec();
        let at = at.index(damaged.len());
        damaged[at] = byte;
        restore_and_use(&damaged)?;
        restore_and_use(&GOLDEN[..cut.index(GOLDEN.len())])?;
    }

    /// The same for bytes that never were a snapshot, with and without a valid header.
    #[test]
    fn arbitrary_bytes_are_refused_or_usable(
        body in proptest::collection::vec(any::<u8>(), 0..256),
    ) {
        restore_and_use(&body)?;
        let mut headed = GOLDEN[..=SNAPSHOT_MAGIC.len()].to_vec();
        headed.extend(&body);
        restore_and_use(&headed)?;
    }
}

/// A toy server holding one plain session with a deferred request.
fn deferred_rig() -> (Rig, TokenId) {
    let mut rig = Rig::strict();
    let token = rig.defer_on(rig.session, &request(toy::WAIT));
    (rig, token)
}

/// Restores `bytes` into a fresh toy server and returns the error.
fn refused(bytes: &[u8]) -> SnapshotError {
    let wakes = Arc::new(Wakes::default());
    let mut server = toy_server(UnknownPolicy::Strict, &wakes);
    let before = server.save().unwrap();
    let error = server.restore(bytes).unwrap_err();
    // A refused snapshot leaves the server as it was.
    assert_eq!(server.save().unwrap(), before);
    error
}

#[test]
fn bytes_that_are_not_this_builds_snapshot_are_refused() {
    let (rig, _) = deferred_rig();
    let saved = rig.server.save().unwrap();
    assert!(restores(&saved));

    let mut other_kind = saved.clone();
    other_kind[0] ^= 0xFF;
    assert_eq!(refused(&other_kind), SnapshotError::BadMagic);

    // The version is the byte after the magic.
    let mut other_version = saved.clone();
    other_version[SNAPSHOT_MAGIC.len()] += 1;
    assert_eq!(
        refused(&other_version),
        SnapshotError::UnsupportedVersion {
            found: SNAPSHOT_VERSION + 1
        }
    );

    let mut long = saved.clone();
    long.push(0);
    assert_eq!(refused(&long), SnapshotError::TrailingBytes { count: 1 });

    let truncated = &saved[..saved.len() - 1];
    assert_eq!(
        refused(truncated),
        SnapshotError::Decode(postcard::Error::DeserializeUnexpectedEnd)
    );
    assert_eq!(
        refused(&[]),
        SnapshotError::Decode(postcard::Error::DeserializeUnexpectedEnd)
    );
}

/// Whether `bytes` restore into a fresh toy server.
fn restores(bytes: &[u8]) -> bool {
    let wakes = Arc::new(Wakes::default());
    toy_server(UnknownPolicy::Strict, &wakes)
        .restore(bytes)
        .is_ok()
}

#[test]
fn a_snapshot_of_another_server_is_refused() {
    let (rig, _) = deferred_rig();
    let saved = rig.server.save().unwrap();
    let mut other = Server::new("other", UnknownPolicy::Strict)
        .with_interfaces(toy_interfaces())
        .with_family(Tally::default());
    assert_eq!(
        other.restore(&saved),
        Err(SnapshotError::WrongServer {
            expected: "other".to_owned(),
            found: "toy".to_owned()
        })
    );
}

/// An interface with the name of the toy's child and a different state.
mod impostor {
    use serde::{Deserialize, Serialize};

    use crate::commands;

    #[derive(Serialize, Deserialize)]
    pub(super) struct Child {
        pub(super) a: u64,
        pub(super) b: [u8; 9],
    }

    commands! {
        impl Child {}
    }
}

#[test]
fn a_session_is_restored_only_as_an_interface_the_server_lists() {
    let mut rig = Rig::strict();
    rig.open_child(0xC1);
    let saved = rig.server.save().unwrap();
    let restore_with = |interfaces: Interfaces| {
        let mut server = Server::new("toy", UnknownPolicy::Strict)
            .with_interfaces(interfaces)
            .with_family(Tally::default());
        server.restore(&saved)
    };
    let toy_only = Interfaces::new().with::<Toy>();
    assert_eq!(
        restore_with(toy_only.clone()),
        Err(SnapshotError::UnknownInterface("Child".to_owned()))
    );
    // A type of the right name whose state is laid out differently.
    let wrong_child = toy_only.with::<impostor::Child>();
    assert_eq!(
        restore_with(wrong_child),
        Err(SnapshotError::State {
            interface: "Child".to_owned(),
            error: postcard::Error::DeserializeUnexpectedEnd
        })
    );
    assert!(restore_with(toy_interfaces()).is_ok());
    // Listing a type twice changes nothing.
    assert!(restore_with(toy_interfaces().with::<Toy>().with::<super::Child>()).is_ok());
    // Both children listed: the saved one could be either, so it is neither.
    let both = toy_interfaces().with::<impostor::Child>();
    assert_eq!(
        restore_with(both.clone()),
        Err(SnapshotError::DuplicateInterface("Child"))
    );
    let mut ambiguous = Server::new("toy", UnknownPolicy::Strict).with_interfaces(both);
    let session = rig.kernel.new_object();
    let child = Box::new(impostor::Child { a: 1, b: [2; 9] });
    ambiguous.open(session, child).unwrap();
    assert_eq!(
        ambiguous.save(),
        Err(SnapshotError::DuplicateInterface("Child"))
    );
    assert_eq!(
        SnapshotError::DuplicateInterface("Child").to_string(),
        "two interface types are named \"Child\""
    );

    // And no snapshot is written that the server itself could not restore.
    let mut unlisted = Server::new("toy", UnknownPolicy::Strict);
    let session = rig.kernel.new_object();
    unlisted.open(session, Box::new(Mute)).unwrap();
    assert_eq!(
        unlisted.save(),
        Err(SnapshotError::UnknownInterface("Mute".to_owned()))
    );
}

/// Family state that is not the toy's.
#[derive(Serialize, Deserialize)]
struct Wide {
    a: u64,
    b: u64,
}

#[test]
fn family_state_is_restored_only_into_a_server_that_has_some() {
    let (rig, _) = deferred_rig();
    let with_family = rig.server.save().unwrap();
    let bare = || Server::new("toy", UnknownPolicy::Strict).with_interfaces(toy_interfaces());
    assert_eq!(
        bare().restore(&with_family),
        Err(SnapshotError::FamilyMismatch)
    );
    let without_family = bare().save().unwrap();
    assert_eq!(refused(&without_family), SnapshotError::FamilyMismatch);
    assert_eq!(bare().restore(&without_family), Ok(()));

    // A family of another shape cannot load the state.
    let mut wide = bare().with_family(Wide { a: 1, b: 2 });
    assert_eq!(
        wide.restore(&with_family),
        Err(SnapshotError::Decode(
            postcard::Error::DeserializeUnexpectedEnd
        ))
    );
    assert_eq!(wide.family::<Wide>().map(|w| (w.a, w.b)), Some((1, 2)));
}

#[test]
fn a_snapshot_whose_requests_disagree_is_refused() {
    type Damage = fn(&mut Rig, TokenId);
    let cases: [(Damage, &str); 5] = [
        (
            |rig, _| {
                let stranger = rig.kernel.new_object();
                rig.server.queue.submit(stranger).unwrap();
            },
            "the queue names a session that is not open",
        ),
        (
            |rig, token| rig.server.next_token = token.raw(),
            "a pending token was never handed out",
        ),
        (
            |rig, token| {
                let other = rig.second_session();
                let request = Pending {
                    session: other,
                    framing: Framing::Cmif,
                    command: 0,
                    request_size: 0,
                };
                rig.server.pending.insert(token, request);
            },
            "a token is pending for a session with no deferred request",
        ),
        (
            |rig, token| {
                let request = rig.server.pending[&token];
                rig.server.next_token += 1;
                rig.server.pending.insert(TokenId(token.raw() + 1), request);
            },
            "two tokens are pending for one session",
        ),
        (
            |rig, _| rig.server.pending.clear(),
            "a deferred request has no pending token",
        ),
    ];
    for (damage, message) in cases {
        let (mut rig, token) = deferred_rig();
        damage(&mut rig, token);
        // A server in that state is not saved,
        let inconsistent = SnapshotError::Inconsistent(message);
        assert_eq!(rig.server.save(), Err(inconsistent.clone()));
        // and the bytes it would have written are not restored.
        let saved = rig.server.encode_state().unwrap();
        assert_eq!(refused(&saved), inconsistent);
        drop(parked());
    }
    // The token id just below the next one is the newest that can be pending.
    let (rig, token) = deferred_rig();
    assert_eq!(rig.server.next_token, token.raw() + 1);
    assert!(restores(&rig.server.save().unwrap()));
    drop(parked());
}

#[test]
fn a_server_is_not_saved_between_taking_a_completion_and_delivering_it() {
    let (mut rig, token) = deferred_rig();
    let session = rig.session;
    parked_one().fail(ResultCode::SUCCESS);

    // Posted and not taken: the server is saved, with the request as still deferred. The
    // completion is not in the snapshot, so a server restored from it has nothing to deliver
    // and the work is resumed with a new token.
    let posted = rig.server.save().unwrap();
    let wakes = Arc::new(Wakes::default());
    let mut restored = toy_server(UnknownPolicy::Strict, &wakes);
    assert_eq!(restored.restore(&posted), Ok(()));
    assert_eq!(restored.pending().collect::<Vec<_>>(), [(token, session)]);
    assert!(restored.queue().is_deferred(session));
    assert!(restored.take_completed().is_none());
    assert!(restored.reissue(token).is_some());
    drop(parked());

    // Taken and not delivered: the request is neither pending nor answered, which no snapshot
    // holds. Saving is refused, and the bytes the server would have written are not restored.
    let completed = rig.server.take_completed().unwrap();
    let in_between = SnapshotError::Inconsistent("a deferred request has no pending token");
    assert_eq!(rig.server.save(), Err(in_between.clone()));
    assert_eq!(refused(&rig.server.encode_state().unwrap()), in_between);

    // Delivered: the server is saved again, and the snapshot restores.
    let mut buffer = vec![0; BUFFER_SIZE];
    let delivered = rig.server.deliver(&mut rig.kernel, completed, &mut buffer);
    assert_eq!(delivered, Ok(()));
    assert!(restores(&rig.server.save().unwrap()));
}

/// The scene with the framing of one of its deferred requests changed, as bytes.
fn reframed(request: usize, framing: Framing) -> Vec<u8> {
    let mut scene = scene();
    let token = scene.tokens[request];
    scene.rig.server.pending.get_mut(&token).unwrap().framing = framing;
    drop(parked());
    scene.rig.server.encode_state().unwrap()
}

#[test]
fn a_deferred_request_is_framed_as_its_session_is() {
    // The scene's requests, by index: 0 on a plain session (CMIF), 1 on a domain, 2 on a plain
    // session (TIPC).
    let not_a_domain = "a deferred request has a domain reply on a session that is not a domain";
    let no_domain_header =
        "a deferred request on a domain session has a reply that is not a domain's";
    let tipc_on_domain = "a deferred TIPC request is on a domain session";
    let cases = [
        (0, Framing::Domain, not_a_domain),
        (2, Framing::Domain, not_a_domain),
        (1, Framing::Cmif, no_domain_header),
        (1, Framing::Tipc(toy::TIPC_WAIT), tipc_on_domain),
    ];
    for (request, framing, message) in cases {
        let error = refused(&reframed(request, framing));
        assert_eq!(error, SnapshotError::Inconsistent(message), "{request}");
    }
    // Each request in a framing its session can have is accepted: its own, and on a plain
    // session the other of CMIF and TIPC, which a snapshot cannot tell apart from its own.
    for (request, framing) in [
        (0, Framing::Cmif),
        (0, Framing::Tipc(toy::TIPC_WAIT)),
        (1, Framing::Domain),
        (2, Framing::Tipc(toy::TIPC_WAIT)),
        (2, Framing::Cmif),
    ] {
        assert!(restores(&reframed(request, framing)), "{request}");
    }

    // A server holding such a request is not saved either.
    let mut scene = scene();
    let token = scene.tokens[0];
    scene.rig.server.pending.get_mut(&token).unwrap().framing = Framing::Domain;
    assert_eq!(
        scene.rig.server.save(),
        Err(SnapshotError::Inconsistent(not_a_domain))
    );
    drop(parked());
}

#[test]
fn a_snapshot_is_restored_only_into_a_server_with_as_many_workers() {
    let (mut rig, _) = deferred_rig();
    let session = rig.session;
    // The same queue on one worker more than a server has.
    let mut wider = RequestQueue::with_workers(DEFAULT_WORKERS + 1).unwrap();
    assert_eq!(wider.submit(session), Ok(Admission::Start));
    wider.defer(session).unwrap();
    assert_eq!(rig.server.queue().workers(), DEFAULT_WORKERS);
    assert!(restores(&rig.server.encode_state().unwrap()));
    rig.server.queue = wider;
    assert_eq!(
        refused(&rig.server.encode_state().unwrap()),
        SnapshotError::Inconsistent("the queue's count of workers is not the server's")
    );
    drop(parked());
}

#[test]
fn a_row_nothing_refers_to_is_refused_whatever_its_count() {
    let mut rig = Rig::strict();
    let child = rig.open_child(0xC1);
    assert!(restores(&rig.server.encode_state().unwrap()));
    // The child's session goes and its row stays.
    let orphan = rig.server.sessions.remove(&child).unwrap();
    assert_eq!(
        refused(&rig.server.encode_state().unwrap()),
        SnapshotError::Inconsistent("a reference count is not the number of references")
    );
    // With its count set to the number of references it has, none, it is still not a row a
    // server can hold: nothing could ever release it.
    rig.server.targets.set_refs(orphan, 0);
    assert_eq!(
        refused(&rig.server.encode_state().unwrap()),
        SnapshotError::Inconsistent("a target has a reference count of zero")
    );
}
