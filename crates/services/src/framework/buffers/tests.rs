//! The check of a request's descriptors against a declaration, and which descriptor each declared
//! buffer is.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{
    BUFFER_SIZE, BufferDescriptor, CommandType, Message, ReceiveDescriptor, RequestBuilder,
    SessionKind, StaticDescriptor,
};

use super::*;

/// How many descriptors of each list a request carries, in the header's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Counts {
    x: usize,
    a: usize,
    b: usize,
    w: usize,
    c: usize,
}

impl Counts {
    fn get(mut self, list: DescriptorList) -> usize {
        *self.get_mut(list)
    }

    fn get_mut(&mut self, list: DescriptorList) -> &mut usize {
        match list {
            DescriptorList::X => &mut self.x,
            DescriptorList::A => &mut self.a,
            DescriptorList::B => &mut self.b,
            DescriptorList::W => &mut self.w,
            DescriptorList::C => &mut self.c,
        }
    }
}

/// Every list, in the order the check compares them.
const LISTS: [DescriptorList; 5] = [
    DescriptorList::X,
    DescriptorList::A,
    DescriptorList::B,
    DescriptorList::W,
    DescriptorList::C,
];

/// The bytes of a request carrying `counts` descriptors, each with its own address.
fn request_with(counts: Counts) -> Vec<u8> {
    let mut build = RequestBuilder::cmif(CommandType::Request, 0);
    let address = |list: u64, i: usize| (list << 32) | u64::try_from(i).unwrap();
    let buffer = |address| BufferDescriptor {
        address,
        size: 0x10,
        flags: 0,
    };
    for i in 0..counts.x {
        build = build.static_descriptor(StaticDescriptor {
            counter: 0,
            address: address(1, i),
            size: 0x10,
        });
    }
    for i in 0..counts.a {
        build = build.send_buffer(buffer(address(2, i)));
    }
    for i in 0..counts.b {
        build = build.receive_buffer(buffer(address(3, i)));
    }
    for i in 0..counts.w {
        build = build.exchange_buffer(buffer(address(4, i)));
    }
    for i in 0..counts.c {
        build = build.receive_descriptor(ReceiveDescriptor {
            address: address(5, i),
            size: 0x10,
        });
    }
    let mut bytes = vec![0; BUFFER_SIZE];
    build.build(&mut bytes).unwrap();
    bytes
}

/// What `check` says of a CMIF request carrying `counts` descriptors.
fn check_counts(declared: &[DeclaredBuffer], counts: Counts) -> Result<(), Fault> {
    check_counts_over(Protocol::Cmif, declared, counts)
}

/// What `check` says of a request carrying `counts` descriptors, as if it came in `protocol`.
fn check_counts_over(
    protocol: Protocol,
    declared: &[DeclaredBuffer],
    counts: Counts,
) -> Result<(), Fault> {
    let bytes = request_with(counts);
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    check(protocol, declared, &message)
}

fn buffer(direction: Direction, kind: BufferKind) -> DeclaredBuffer {
    DeclaredBuffer { direction, kind }
}

fn count(list: DescriptorList, sent: usize, declared: usize) -> Result<(), Fault> {
    Err(Fault::DescriptorCount {
        list,
        sent,
        declared,
    })
}

/// Where a client put one buffer: its index in the A (or B) list and in the X (or C) list, where
/// it has one.
type ClientPlace = (Option<usize>, Option<usize>);

/// Where a client puts each declared buffer: libnx's `_serviceRequestProcessBuffer`, which sends
/// a map-alias buffer as the next A (or B) descriptor, a pointer buffer as the next X (or C)
/// descriptor, and an auto-select buffer as one of each (`cmifRequestInAutoBuffer`,
/// `cmifRequestOutAutoBuffer`). Returns, per buffer, its index in the A/B list and in the X/C
/// list where it has one, and the request's counts.
fn as_a_client_sends(declared: &[DeclaredBuffer]) -> (Vec<ClientPlace>, Counts) {
    let mut counts = Counts::default();
    let places = declared
        .iter()
        .map(|b| {
            let (map_alias_list, pointer_list) = match b.direction {
                Direction::In => (DescriptorList::A, DescriptorList::X),
                Direction::Out => (DescriptorList::B, DescriptorList::C),
            };
            let mut take = |list| {
                let slot = counts.get_mut(list);
                let index = *slot;
                *slot += 1;
                index
            };
            match b.kind {
                BufferKind::MapAlias => (Some(take(map_alias_list)), None),
                BufferKind::Pointer => (None, Some(take(pointer_list))),
                BufferKind::AutoSelect => {
                    let pointer = take(pointer_list);
                    (Some(take(map_alias_list)), Some(pointer))
                }
            }
        })
        .collect();
    (places, counts)
}

#[test]
fn each_kind_accepts_its_own_descriptors_and_refuses_the_others() {
    use BufferKind::{AutoSelect, MapAlias, Pointer};
    use DescriptorList::{A, B, C, X};
    use Direction::{In, Out};
    // What a request offers for the one buffer: its map-alias descriptor, its pointer
    // descriptor, both, or neither.
    let offers = |direction| {
        let (map_alias, pointer) = match direction {
            In => (
                Counts {
                    a: 1,
                    ..Counts::default()
                },
                Counts {
                    x: 1,
                    ..Counts::default()
                },
            ),
            Out => (
                Counts {
                    b: 1,
                    ..Counts::default()
                },
                Counts {
                    c: 1,
                    ..Counts::default()
                },
            ),
        };
        let both = Counts {
            x: pointer.x,
            a: map_alias.a,
            b: map_alias.b,
            c: pointer.c,
            w: 0,
        };
        [map_alias, pointer, both, Counts::default()]
    };
    let ok = Ok(());
    // Expected answers, in the order of `offers`: map-alias, pointer, both, neither.
    let table = [
        (
            In,
            MapAlias,
            [ok, count(X, 1, 0), count(X, 1, 0), count(A, 0, 1)],
        ),
        (
            In,
            Pointer,
            [count(X, 0, 1), ok, count(A, 1, 0), count(X, 0, 1)],
        ),
        (
            In,
            AutoSelect,
            [count(X, 0, 1), count(A, 0, 1), ok, count(X, 0, 1)],
        ),
        // An extra C descriptor is accepted, as on the console (module docs).
        (Out, MapAlias, [ok, count(B, 0, 1), ok, count(B, 0, 1)]),
        (
            Out,
            Pointer,
            [count(B, 1, 0), ok, count(B, 1, 0), count(C, 0, 1)],
        ),
        (
            Out,
            AutoSelect,
            [count(C, 0, 1), count(B, 0, 1), ok, count(B, 0, 1)],
        ),
    ];
    for (direction, kind, expected) in table {
        let declared = [buffer(direction, kind)];
        for (offer, expected) in offers(direction).into_iter().zip(expected) {
            assert_eq!(
                check_counts(&declared, offer),
                expected,
                "{direction:?} {kind:?} offered {offer:?}"
            );
        }
    }
}

#[test]
fn a_declaration_of_nothing_refuses_every_descriptor_but_c() {
    assert_eq!(check_counts(&[], Counts::default()), Ok(()));
    for list in LISTS {
        let mut counts = Counts::default();
        *counts.get_mut(list) = 1;
        let expected = if list == DescriptorList::C {
            Ok(())
        } else {
            count(list, 1, 0)
        };
        assert_eq!(check_counts(&[], counts), expected, "{list:?}");
    }
}

#[test]
fn a_w_descriptor_is_refused_whatever_is_declared() {
    let declared = [
        buffer(Direction::In, BufferKind::AutoSelect),
        buffer(Direction::Out, BufferKind::AutoSelect),
    ];
    let (_, mut counts) = as_a_client_sends(&declared);
    assert_eq!(check_counts(&declared, counts), Ok(()));
    counts.w = 1;
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::W, 1, 0)
    );
}

/// Divergence from the console, which refuses a map-alias descriptor whose transfer mode the
/// command does not allow: the mode is not looked at, as in Eden. Every value of the 2-bit field
/// passes (libnx `hipc.h`, `HipcBufferMode`: `Normal` 0, `NonSecure` 1, `Invalid` 2,
/// `NonDevice` 3).
#[test]
fn the_transfer_mode_is_not_checked() {
    let declared = [
        buffer(Direction::In, BufferKind::MapAlias),
        buffer(Direction::Out, BufferKind::MapAlias),
    ];
    for flags in 0..4 {
        let descriptor = BufferDescriptor {
            address: 0x1000,
            size: 0x10,
            flags,
        };
        let mut bytes = vec![0; BUFFER_SIZE];
        RequestBuilder::cmif(CommandType::Request, 0)
            .send_buffer(descriptor)
            .receive_buffer(descriptor)
            .build(&mut bytes)
            .unwrap();
        let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
        assert_eq!(message.send_buffers[0].flags, flags);
        assert_eq!(
            check(Protocol::Cmif, &declared, &message),
            Ok(()),
            "mode {flags}"
        );
    }
}

#[test]
fn the_counts_are_compared_in_header_order() {
    // Every list wrong at once: each is reported only once the lists before it are right.
    let declared = [buffer(Direction::Out, BufferKind::Pointer)];
    let mut counts = Counts {
        x: 1,
        a: 2,
        b: 3,
        w: 4,
        c: 0,
    };
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::X, 1, 0)
    );
    counts.x = 0;
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::A, 2, 0)
    );
    counts.a = 0;
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::B, 3, 0)
    );
    counts.b = 0;
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::W, 4, 0)
    );
    counts.w = 0;
    assert_eq!(
        check_counts(&declared, counts),
        count(DescriptorList::C, 0, 1)
    );
    counts.c = 1;
    assert_eq!(check_counts(&declared, counts), Ok(()));
}

#[test]
fn c_descriptors_must_cover_the_declared_buffers_and_may_be_more() {
    let declared = [
        buffer(Direction::Out, BufferKind::Pointer),
        buffer(Direction::Out, BufferKind::AutoSelect),
    ];
    let (_, counts) = as_a_client_sends(&declared);
    let with_c = |c| Counts { c, ..counts };
    // One short of the two declared, exactly two, and one more.
    assert_eq!(
        check_counts(&declared, with_c(1)),
        count(DescriptorList::C, 1, 2)
    );
    assert_eq!(check_counts(&declared, with_c(2)), Ok(()));
    assert_eq!(check_counts(&declared, with_c(3)), Ok(()));
}

/// Divergence from CMIF, after the console's TIPC header check (module docs, "TIPC"): on TIPC
/// the C descriptors must be exactly as many as the declared out pointer and auto-select
/// buffers, so a command that declares none refuses a single one.
#[test]
fn c_descriptors_on_tipc_must_match_exactly() {
    let tipc =
        |declared: &[DeclaredBuffer], counts| check_counts_over(Protocol::Tipc, declared, counts);
    let with_c = |counts: Counts, c| Counts { c, ..counts };
    // Nothing declared: none is fine, one is refused over TIPC and accepted over CMIF.
    assert_eq!(tipc(&[], Counts::default()), Ok(()));
    assert_eq!(
        tipc(&[], with_c(Counts::default(), 1)),
        count(DescriptorList::C, 1, 0)
    );
    assert_eq!(check_counts(&[], with_c(Counts::default(), 1)), Ok(()));
    // Two declared: one short, exactly two, and one more.
    let declared = [
        buffer(Direction::Out, BufferKind::Pointer),
        buffer(Direction::Out, BufferKind::AutoSelect),
    ];
    let (_, counts) = as_a_client_sends(&declared);
    assert_eq!(
        tipc(&declared, with_c(counts, 1)),
        count(DescriptorList::C, 1, 2)
    );
    assert_eq!(tipc(&declared, with_c(counts, 2)), Ok(()));
    assert_eq!(
        tipc(&declared, with_c(counts, 3)),
        count(DescriptorList::C, 3, 2)
    );
    // The other lists are compared before C, as on CMIF.
    let both = Counts {
        a: 1,
        ..with_c(counts, 3)
    };
    assert_eq!(tipc(&declared, both), count(DescriptorList::A, 1, 0));
}

#[test]
fn each_list_is_numbered_on_its_own() {
    use BufferKind::{AutoSelect, MapAlias, Pointer};
    use Direction::{In, Out};
    let declared = [
        buffer(In, Pointer),
        buffer(Out, MapAlias),
        buffer(In, MapAlias),
        buffer(In, AutoSelect),
        buffer(Out, Pointer),
        buffer(Out, AutoSelect),
        buffer(In, Pointer),
    ];
    let place = |kind, map_alias, pointer| {
        Some(Place {
            kind,
            map_alias,
            pointer,
        })
    };
    // Eden would read in buffer 1 from A descriptor 1 and in buffer 3 from X descriptor 3.
    assert_eq!(locate(&declared, In, 0), place(Pointer, 0, 0));
    assert_eq!(locate(&declared, In, 1), place(MapAlias, 0, 1));
    assert_eq!(locate(&declared, In, 2), place(AutoSelect, 1, 1));
    assert_eq!(locate(&declared, In, 3), place(Pointer, 2, 2));
    assert_eq!(locate(&declared, In, 4), None);
    assert_eq!(locate(&declared, Out, 0), place(MapAlias, 0, 0));
    assert_eq!(locate(&declared, Out, 1), place(Pointer, 1, 0));
    assert_eq!(locate(&declared, Out, 2), place(AutoSelect, 1, 1));
    assert_eq!(locate(&declared, Out, 3), None);
    assert_eq!(locate(&[], In, 0), None);
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x4255_4646_4552_5331),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

fn any_buffer() -> impl Strategy<Value = DeclaredBuffer> {
    let direction = prop_oneof![Just(Direction::In), Just(Direction::Out)];
    let kind = prop_oneof![
        Just(BufferKind::MapAlias),
        Just(BufferKind::Pointer),
        Just(BufferKind::AutoSelect),
    ];
    (direction, kind).prop_map(|(direction, kind)| DeclaredBuffer { direction, kind })
}

proptest! {
    #![proptest_config(config())]

    /// Whatever a command declares (up to libnx's eight buffers, `attr0` to `attr7`), the request
    /// a client sends for it passes, and one descriptor more or fewer in any list is refused
    /// naming that list; a C descriptor more is accepted.
    #[test]
    fn a_clients_request_passes_and_any_list_off_by_one_is_named(
        declared in prop::collection::vec(any_buffer(), 0..=8),
    ) {
        let (_, sent) = as_a_client_sends(&declared);
        prop_assert_eq!(check_counts(&declared, sent), Ok(()));
        for list in LISTS {
            let declared_count = sent.get(list);
            let mut more = sent;
            *more.get_mut(list) += 1;
            let expected = if list == DescriptorList::C {
                Ok(())
            } else {
                count(list, declared_count + 1, declared_count)
            };
            prop_assert_eq!(check_counts(&declared, more), expected, "{:?} more", list);
            if declared_count > 0 {
                let mut fewer = sent;
                *fewer.get_mut(list) -= 1;
                let expected = count(list, declared_count - 1, declared_count);
                prop_assert_eq!(check_counts(&declared, fewer), expected, "{:?} fewer", list);
            }
        }
    }

    /// Each declared buffer is located where the client put it.
    #[test]
    fn a_buffer_is_located_where_a_client_puts_it(
        declared in prop::collection::vec(any_buffer(), 0..=8),
    ) {
        let (places, _) = as_a_client_sends(&declared);
        for direction in [Direction::In, Direction::Out] {
            let mine = declared.iter().zip(&places).filter(|(b, _)| b.direction == direction);
            let mut last = 0;
            for (index, (b, (map_alias, pointer))) in mine.enumerate() {
                let place = locate(&declared, direction, index).unwrap();
                prop_assert_eq!(place.kind, b.kind);
                if let Some(map_alias) = map_alias {
                    prop_assert_eq!(place.map_alias, *map_alias);
                }
                if let Some(pointer) = pointer {
                    prop_assert_eq!(place.pointer, *pointer);
                }
                last = index + 1;
            }
            prop_assert_eq!(locate(&declared, direction, last), None);
        }
    }
}
