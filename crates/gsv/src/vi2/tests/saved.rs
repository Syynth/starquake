//! The session's saved form: its golden bytes, each relation a restore refuses with its exact
//! error, each limit at its edge, and a restored server used through its commands.

use serde::Deserialize;

use super::*;
use crate::vi2::saved::{SavedVi2Session, Vi2SessionError};

/// `"#ld"`, a known display, as a monitor's name: a three-byte name, so its varint is four
/// bytes long (`0xa3, 0xd8, 0x91, 0x03`).
const LD: u64 = u64::from_le_bytes(*b"#ld\0\0\0\0\0");

fn object(index: u32, generation: u32) -> ObjectId {
    ObjectId::from_parts(index, generation)
}

/// A session whose every field holds a distinct value where a valid session allows it.
fn distinct() -> Vi2Session {
    let mut notifications = [0; NOTIFICATION_WORDS];
    notifications[0] = 0x05;
    let mut groups = vec![RequestGroup::default(); 8];
    groups[0] = RequestGroup {
        event: Some(object(0x06, 0x07)),
        monitor: Some(Monitor {
            display: LD,
            kind: 0x02,
        }),
    };
    groups[1] = RequestGroup {
        event: Some(object(0x08, 0x09)),
        monitor: None,
    };
    Vi2Session {
        notification_event: Some(object(0x01, 0x02)),
        notifications,
        groups,
        group_count: Some(0x04),
        display_references: 0x0A,
        clients: vec![Client {
            token: 0x0B,
            retire: 0x0C,
            available_bit: 0x0D,
            bound: true,
            layer: Some(0x0E),
            available: 0x0F,
            levels: Some([0x10, 0x11, 0x12]),
        }],
        endpoint: Some(0x0B),
        next_layer: 0x13,
    }
}

/// The layout (`vi2::saved`, "Layout"), byte for byte: `Option` tags, varints, the bitmap with
/// no length, the groups and clients with theirs.
#[test]
fn the_saved_session_layout_is_pinned() {
    let mut expected = vec![
        1, 0x01, 0x02, // notification event: Some(index, generation)
        0x05, // bitmap word 0
    ];
    expected.extend([0; NOTIFICATION_WORDS - 1]); // bitmap words 1 to 31
    expected.extend([
        8, // eight groups
        1, 0x06, 0x07, 1, 0xA3, 0xD8, 0x91, 0x03, 0x02, // group 0: event; monitor (#ld, 2)
        1, 0x08, 0x09, 0, // group 1: event; no monitor
    ]);
    expected.extend([0; 12]); // groups 2 to 7: no event, no monitor
    expected.extend([
        1, 0x04, // group count
        0x0A, // display references
        1,    // one client
        0x0B, 0x0C, 0x0D, 1, // token, retire, available bit, bound
        1, 0x0E, // layer
        0x0F, // free slots
        1, 0x10, 0x11, 0x12, // levels
        1, 0x0B, // endpoint
        0x13, // next layer
    ]);
    let session = distinct();
    let bytes = session.save().unwrap();
    assert_eq!(bytes, expected);
    assert_eq!(postcard::from_bytes::<Vi2Session>(&bytes).unwrap(), session);
}

/// What `session` deserialises to through its saved form, and the error if refused.
fn read_back(session: &Vi2Session) -> Result<Vi2Session, Vi2SessionError> {
    let bytes = postcard::to_allocvec(session).unwrap();
    let saved =
        SavedVi2Session::deserialize(&mut postcard::Deserializer::from_bytes(&bytes)).unwrap();
    Vi2Session::try_from(saved)
}

/// `distinct` changed by `change`.
fn with(change: impl FnOnce(&mut Vi2Session)) -> Vi2Session {
    let mut session = distinct();
    change(&mut session);
    session
}

fn client(token: u64, retire: u32, available_bit: u32) -> Client {
    Client::new(token, retire, available_bit)
}

/// A refused case: the session, the error and its words.
type Refusal = (Vi2Session, Vi2SessionError, &'static str);

/// The relations of the groups and the session (`vi2::saved`, steps 1 to 6).
fn session_refusals() -> Vec<Refusal> {
    vec![
        (
            with(|s| s.groups.truncate(7)),
            Vi2SessionError::Groups { found: 7 },
            "a vi2 session holds 7 request groups, not 8",
        ),
        (
            with(|s| s.groups.push(RequestGroup::default())),
            Vi2SessionError::Groups { found: 9 },
            "a vi2 session holds 9 request groups, not 8",
        ),
        (
            with(|s| s.groups[2].monitor = s.groups[0].monitor),
            Vi2SessionError::MonitorWithoutEvent { group: 2 },
            "vi2 request group 2 is registered as a monitor but has no event",
        ),
        (
            with(|s| {
                s.groups[1].monitor = Some(Monitor {
                    display: LD,
                    kind: MONITOR_KINDS,
                });
            }),
            Vi2SessionError::MonitorKind { group: 1, kind: 6 },
            "vi2 request group 1 monitors kind 6, past the last kind",
        ),
        (
            with(|s| {
                s.groups[1].monitor = Some(Monitor {
                    display: 0,
                    kind: 0,
                });
            }),
            Vi2SessionError::MonitorDisplay { group: 1 },
            "vi2 request group 1 monitors a display name no monitor command accepts",
        ),
        (
            with(|s| {
                s.groups[0].monitor = Some(Monitor {
                    display: UNKNOWN,
                    kind: 0,
                });
            }),
            Vi2SessionError::MonitorDisplay { group: 0 },
            "vi2 request group 0 monitors a display name no monitor command accepts",
        ),
        (
            with(|s| s.groups[3].event = s.notification_event),
            Vi2SessionError::NotificationEventShared { group: 3 },
            "vi2 request group 3's event is the notification event",
        ),
        (
            with(|s| s.groups[4].event = s.groups[1].event),
            Vi2SessionError::GroupEventShared {
                first: 1,
                second: 4,
            },
            "vi2 request groups 1 and 4 hold the same event",
        ),
        (
            with(|s| s.group_count = Some(REQUEST_GROUPS)),
            Vi2SessionError::GroupCount { found: 8 },
            "a vi2 session's request group count is 8, not below 8",
        ),
        (
            with(|s| s.display_references = MAX_DISPLAY_REFERENCES + 1),
            Vi2SessionError::DisplayReferences { found: 1025 },
            "a vi2 session's display is open 1025 times, more than 1024",
        ),
        (
            with(|s| s.notifications[1] = 1),
            Vi2SessionError::Notifications { word: 1 },
            "a vi2 session's notification word 1 has a bit raised that nothing raises",
        ),
        (
            with(|s| s.notifications[NOTIFICATION_WORDS - 1] = 0x8000_0000),
            Vi2SessionError::Notifications { word: 31 },
            "a vi2 session's notification word 31 has a bit raised that nothing raises",
        ),
        (
            with(|s| {
                s.next_layer = 0;
                s.clients[0].layer = None;
            }),
            Vi2SessionError::NextLayer { found: 0 },
            "a vi2 session's next layer number is 0, below the first, 1",
        ),
    ]
}

/// The relations of the clients and the endpoint (steps 7 to 9).
fn client_refusals() -> Vec<Refusal> {
    let full: Vec<Client> = (0..17_u32)
        .map(|i| client(u64::from(i) + 1, i % 16 * 2, i % 16 * 2 + 1))
        .collect();
    vec![
        (
            with(|s| {
                s.clients.clone_from(&full);
                s.endpoint = None;
            }),
            Vi2SessionError::TooManyClients { count: 17 },
            "a vi2 session holds 17 clients, more than 16",
        ),
        (
            with(|s| {
                s.clients[0].token = 0;
                s.endpoint = None;
            }),
            Vi2SessionError::ZeroToken { index: 0 },
            "vi2 client 0 has the token zero",
        ),
        (
            with(|s| {
                s.clients.push(client(0x21, 1, 2));
                s.clients.push(client(0x0B, 3, 4));
            }),
            Vi2SessionError::DuplicateToken {
                first: 0,
                second: 2,
            },
            "vi2 clients 0 and 2 have the same token",
        ),
        (
            with(|s| s.clients[0].retire = CLIENT_NOTIFICATIONS),
            Vi2SessionError::ClientBits { index: 0 },
            "vi2 client 0's notification bits are not two different bits below 32",
        ),
        (
            with(|s| s.clients[0].available_bit = CLIENT_NOTIFICATIONS),
            Vi2SessionError::ClientBits { index: 0 },
            "vi2 client 0's notification bits are not two different bits below 32",
        ),
        (
            with(|s| s.clients[0].available_bit = s.clients[0].retire),
            Vi2SessionError::ClientBits { index: 0 },
            "vi2 client 0's notification bits are not two different bits below 32",
        ),
        (
            with(|s| s.clients.push(client(0x21, 0x0D, 1))),
            Vi2SessionError::SharedBit {
                first: 0,
                second: 1,
            },
            "vi2 clients 0 and 1 hold the same notification bit",
        ),
        (
            with(|s| s.clients[0].available = ALL_SLOTS + 1),
            Vi2SessionError::Availability { index: 0 },
            "vi2 client 0 has a free slot past the last",
        ),
        (
            with(|s| s.clients[0].layer = Some(s.next_layer)),
            Vi2SessionError::LayerAhead { index: 0 },
            "vi2 client 0's layer is not one the session has numbered",
        ),
        (
            with(|s| s.clients[0].layer = Some(FIRST_LAYER - 1)),
            Vi2SessionError::LayerAhead { index: 0 },
            "vi2 client 0's layer is not one the session has numbered",
        ),
        (
            with(|s| {
                let mut second = client(0x21, 1, 2);
                second.layer = Some(0x0E);
                s.clients.push(second);
            }),
            Vi2SessionError::SharedLayer {
                first: 0,
                second: 1,
            },
            "vi2 clients 0 and 1 have the same layer",
        ),
        (
            with(|s| s.clients[0].levels = Some([f32::NAN.to_bits(), 0, 0])),
            Vi2SessionError::Levels { index: 0 },
            "vi2 client 0's output levels are not ones the session accepts",
        ),
        (
            with(|s| s.clients[0].levels = Some([0, 0, (-1.0_f32).to_bits()])),
            Vi2SessionError::Levels { index: 0 },
            "vi2 client 0's output levels are not ones the session accepts",
        ),
        (
            with(|s| s.endpoint = Some(0x0C)),
            Vi2SessionError::EndpointWithoutClient,
            "a vi2 session's endpoint names no registered client",
        ),
    ]
}

/// Two faults at once: the earlier check's error.
fn ordering_refusals() -> Vec<Refusal> {
    vec![
        (
            with(|s| {
                s.groups[1].monitor = Some(Monitor {
                    display: LD,
                    kind: MONITOR_KINDS,
                });
                s.group_count = Some(REQUEST_GROUPS);
            }),
            Vi2SessionError::MonitorKind { group: 1, kind: 6 },
            "vi2 request group 1 monitors kind 6, past the last kind",
        ),
        (
            with(|s| {
                s.groups[5].monitor = s.groups[0].monitor;
                s.groups[3].event = s.notification_event;
            }),
            Vi2SessionError::NotificationEventShared { group: 3 },
            "vi2 request group 3's event is the notification event",
        ),
        (
            with(|s| {
                s.display_references = MAX_DISPLAY_REFERENCES + 1;
                s.notifications[2] = 1;
            }),
            Vi2SessionError::DisplayReferences { found: 1025 },
            "a vi2 session's display is open 1025 times, more than 1024",
        ),
        (
            with(|s| {
                s.clients[0].token = 0;
                s.clients[0].retire = CLIENT_NOTIFICATIONS;
            }),
            Vi2SessionError::ZeroToken { index: 0 },
            "vi2 client 0 has the token zero",
        ),
        (
            with(|s| {
                s.clients.push(client(0x21, 1, 2));
                s.clients[1].available = ALL_SLOTS + 1;
                s.clients[0].available = ALL_SLOTS + 1;
            }),
            Vi2SessionError::Availability { index: 0 },
            "vi2 client 0 has a free slot past the last",
        ),
    ]
}

/// Each relation, refused with its exact error and said in plain words; with two faults the
/// earlier check's error comes back.
#[test]
fn each_unreachable_saved_session_is_refused() {
    let cases = [session_refusals(), client_refusals(), ordering_refusals()].concat();
    for (session, error, words) in cases {
        assert_eq!(read_back(&session), Err(error), "{session:?}");
        assert_eq!(error.to_string(), words);
    }
}

/// Each limit exactly at its edge is accepted, as is the state live commands leave: a client
/// bound and holding the endpoint after the display closed, a layer with no display.
#[test]
fn a_saved_session_at_each_edge_is_accepted() {
    let full: Vec<Client> = (0..16_u32)
        .map(|i| client(u64::from(i) + 1, i * 2, i * 2 + 1))
        .collect();
    let cases = [
        with(|s| {
            s.groups[1].monitor = Some(Monitor {
                display: LD,
                kind: MONITOR_KINDS - 1,
            });
        }),
        with(|s| s.group_count = Some(REQUEST_GROUPS - 1)),
        with(|s| s.display_references = MAX_DISPLAY_REFERENCES),
        with(|s| s.notifications[0] = u32::MAX),
        with(|s| s.clients[0].layer = Some(s.next_layer - 1)),
        with(|s| s.clients[0].layer = Some(FIRST_LAYER)),
        with(|s| {
            s.clients.clone_from(&full);
            s.endpoint = Some(16);
        }),
        with(|s| {
            s.clients[0].retire = CLIENT_NOTIFICATIONS - 1;
            s.clients[0].available_bit = 0;
        }),
        with(|s| s.clients[0].available = ALL_SLOTS),
        with(|s| s.clients[0].levels = Some([1.0_f32.to_bits(), 0, f32::MAX.to_bits()])),
        with(|s| s.display_references = 0),
        Vi2Session::new(),
    ];
    for session in cases {
        assert_eq!(read_back(&session), Ok(session.clone()), "{session:?}");
    }
}

/// A server saved mid-session restores into a fresh one, and the restored session answers as
/// the saved one would: its client's free slots, its endpoint taken, its group's event made,
/// its display still open and its notification still raised.
#[test]
fn a_restored_session_carries_on() {
    let mut rig = Rig::new();
    let event = rig.group_event(6);
    rig.bound_client(0xA1, 3, 4);
    rig.ok(&open_endpoint(DEFAULT_DISPLAY, DEFAULT_DISPLAY, 0xA1));
    rig.ok(&register_client(0xA2, 5, 6));
    rig.ok(&client_only(0xDA, 0xA2));
    let saved = rig.server.save().unwrap();

    let shared = fresh_shared();
    let mut restored = server_over(&shared, UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    rig.server = restored;
    rig.shared = shared;
    assert_eq!(
        rig.send(&client_only(0xE4, 0xA1)),
        success(GET_CLIENT_AVAILABILITY, &[], &[0x7FFF, 0])
    );
    assert_eq!(
        rig.send(&open_endpoint(DEFAULT_DISPLAY, DEFAULT_DISPLAY, 0xA1)),
        refused(OPEN_ENDPOINT, KernelError::InvalidState)
    );
    assert_eq!(
        rig.send(&create_group(6)),
        result_only(CREATE_REQUEST_GROUP_EVENT, VI_PERMISSION_DENIED)
    );
    let handle = rig.kernel.insert_handle(event).unwrap().raw();
    let reply = rig.send(&duplicate_group(handle, 6));
    assert_eq!(rig.object(Rig::handed(&reply)), event);
    assert_eq!(
        rig.send(&register_client(0xA3, 4, 7)),
        refused(REGISTER_CLIENT_STATE, KernelError::InvalidState)
    );
    rig.ok(&get_notifications(128));
    assert_eq!(
        rig.kernel.memory(BUFFER, 4).unwrap(),
        (1_u32 << 5).to_le_bytes()
    );
    rig.ok(&close_display(DEFAULT_DISPLAY));
}

/// A server snapshot holding an unreachable session (an endpoint naming no registered client)
/// is refused as a whole, with the framework's error for a session state that does not decode.
/// `postcard` carries no custom message ([`postcard::Error::SerdeDeCustom`]), so the cause is
/// pinned on the same session read back alone: [`Vi2SessionError::EndpointWithoutClient`].
#[test]
fn a_server_snapshot_with_an_unreachable_session_is_refused() {
    let unreachable = with(|s| s.endpoint = Some(0x0C));
    assert_eq!(
        read_back(&unreachable),
        Err(Vi2SessionError::EndpointWithoutClient)
    );
    let mut rig = Rig::new();
    rig.replace(unreachable);
    let saved = rig.server.save().unwrap();
    let mut fresh = server_over(&fresh_shared(), UnknownPolicy::Strict);
    assert_eq!(
        fresh.restore(&saved),
        Err(SnapshotError::State {
            interface: Vi2Session::interface().to_owned(),
            error: postcard::Error::SerdeDeCustom,
        })
    );
    assert_eq!(fresh.session_count(), 0);
}

/// The family saves nothing and loads only nothing.
#[test]
fn the_family_saves_nothing() {
    let mut family = Vi2Family::new(fresh_shared());
    assert_eq!(family.save_state().unwrap(), Vec::<u8>::new());
    assert_eq!(family.load_state(&[]), Ok(()));
    assert_eq!(
        family.load_state(&[0]),
        Err(SnapshotError::Inconsistent("the vi2 family saves no state"))
    );
}
