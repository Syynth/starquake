//! Tests of the present table: the FIFO latch on scripted ticks, the release on the latching
//! tick, the drain reaching frames on its drained values, the clock running while a present
//! waits, the table's bounds, and its snapshot.

use std::collections::BTreeSet;
use std::sync::Arc;

use starquake_gpu::interface::{Completions, GpuCompletion};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::SlabId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_services::framework::UnknownPolicy;
use starquake_trace::EventKind;

use super::super::Shared;
use super::super::vsync::{INTERVAL_NS, VsyncSnapshotError};
use super::*;
use crate::completion::{Completion, CompletionQueue};
use crate::note::{BUFFER_RELEASED, FRAME_LATCHED};
use crate::syncpt::waiters::Wait;
use crate::vi2::vi2_server;

/// The number the tests' gsv records its notes under: not the default, so a hard-coded one
/// fails.
const GSV: ServiceId = ServiceId(0x0657);

/// When the tests' first frame starts the clock: not zero, so a deadline counted from zero
/// fails.
const START: HostNanos = HostNanos(0x0123_4567);

/// The two clients, the second's token above the first's, their numbers in the table (the
/// order the tests register them in) and their bits.
const FIRST_TOKEN: u64 = 0x1111_0000_0000_0001;
const SECOND_TOKEN: u64 = 0x2222_0000_0000_0002;
const FIRST: u8 = 0;
const SECOND: u8 = 1;
const FIRST_BIT: u32 = 3;
const SECOND_BIT: u32 = 9;

/// The syncpoint the tests' fences are on.
fn fenced_on() -> SyncpointId {
    SyncpointId::new(0x17).unwrap()
}

/// Event `n`: not generation 0 or 1, so an id built with another generation differs.
fn event(n: u32) -> ObjectId {
    ObjectId::from_parts(n, 5)
}

/// Tick `n`'s deadline, counted from the clock's start at [`START`].
fn tick_at(n: u64) -> HostNanos {
    HostNanos(START.0 + n * INTERVAL_NS)
}

fn fresh() -> Shared {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    Shared::new(Arc::new(Syncpoints::new(completions)), queue, GSV)
}

/// One call a [`Journal`] host received.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Record(EventKind),
    Signal(ObjectId),
}

/// A [`TimerHost`] that keeps every call in one list, in order, and refuses to signal the
/// events in `refused`.
#[derive(Default)]
struct Journal {
    calls: Vec<Call>,
    refused: BTreeSet<ObjectId>,
}

impl TimerHost for Journal {
    fn record(&mut self, kind: EventKind) {
        self.calls.push(Call::Record(kind));
    }

    fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
        unreachable!("the present table unlocks nothing")
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.calls.push(Call::Signal(event));
        if self.refused.contains(&event) {
            Err(KernelError::InvalidHandle)
        } else {
            Ok(())
        }
    }
}

/// The note calls among `calls`, decoded, and the signals, in order.
fn notes(calls: &[Call]) -> Vec<Note> {
    calls
        .iter()
        .filter_map(|call| match call {
            Call::Record(EventKind::ServiceNote {
                service,
                code,
                value,
            }) if *service == GSV => Some(Note::decode(*code, *value).unwrap()),
            _ => None,
        })
        .collect()
}

/// A request whose every field is distinct and derived from `n`.
fn request(n: u32) -> PresentRequest {
    PresentRequest {
        image: ImageDesc {
            address: 0x10_0000 * u64::from(n),
            width: 100 + n,
            height: 200 + n,
            pitch: 300 + n,
            block_height_log2: 7,
            format: 1,
        },
        shown: Rect {
            x: 0,
            y: 0,
            width: 10 + n,
            height: 20 + n,
        },
        layer: LayerId(u64::from(n) + 40),
        display: DisplayId(u64::from(n) + 50),
    }
}

/// A buffer-queue frame of `slot` showing image `0x100 + slot`, fenced at `value` if given.
fn frame(slot: u8, value: Option<u32>) -> NewFrame {
    NewFrame {
        slot,
        image: 0x100 + u32::from(slot),
        request: request(u32::from(slot)),
        fence: value.map(|value| Fence {
            id: fenced_on(),
            value,
        }),
        holds_slot: true,
    }
}

/// Registers `token` with bit `bit` and event `notify`, which the table gives `number`.
fn register(shared: &Shared, number: u8, token: u64, bit: u32, notify: Option<ObjectId>) {
    let given = shared
        .lock()
        .presents_mut()
        .register(token, bit, notify)
        .unwrap();
    assert_eq!(given, number);
}

/// Queues `new` for client `client`, starting a stopped clock at [`START`], and pushes the
/// change the commit hands back, as 229 does.
fn queue(shared: &Shared, client: u8, new: NewFrame) {
    let mut state = shared.lock();
    let prepared = state.prepare_frame(client, new).unwrap();
    if let Some(change) = state.commit_frame(prepared, Some(START), None) {
        shared.queue().push(Completion::Gpu(change));
    }
}

/// One live vsync turn at `now`: the calls it made and the presents it returned.
fn turn(shared: &Shared, now: HostNanos) -> (Vec<Call>, Vec<PresentRequest>) {
    let mut host = Journal::default();
    let presents = shared.tick_vsync(&mut host, now);
    (host.calls, presents)
}

/// Pushes the drained change `value` of the fence syncpoint and drains it.
fn drain_change(shared: &Shared, value: u32) {
    shared
        .queue()
        .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
            id: fenced_on(),
            value,
        }));
    shared.drain(&mut Journal::default());
}

/// Client [`FIRST`]'s `FrameLatched` of `slot`.
fn latched(slot: u8) -> Note {
    latched_by(FIRST, slot)
}

/// Client `client`'s `FrameLatched` of `slot`.
fn latched_by(client: u8, slot: u8) -> Note {
    Note::FrameLatched {
        client,
        slot,
        image: 0x100 + u32::from(slot),
    }
}

/// Client [`FIRST`]'s `BufferReleased` of `slot`.
fn released(slot: u8) -> Note {
    released_by(FIRST, slot)
}

/// Client `client`'s `BufferReleased` of `slot`.
fn released_by(client: u8, slot: u8) -> Note {
    Note::BufferReleased {
        client,
        slot,
        image: 0x100 + u32::from(slot),
    }
}

/// The reference values written out: PR's fifteen slots (`slot < 15`), its four zero reply
/// words, and the note codes this slice writes.
#[test]
fn the_constants_are_the_references_values() {
    assert_eq!(SLOTS, 15);
    assert_eq!(ALL_SLOTS, (1 << SLOTS) - 1);
    assert_eq!(REPLY_ZERO_WORDS, 4);
    assert_eq!(MAX_PRESENT_CLIENTS, 128);
    assert_eq!((FRAME_LATCHED, BUFFER_RELEASED), (5, 6));
    assert_eq!(SNAPSHOT_MAGIC, *b"SQPR");
    assert_eq!(slot_bit(0), 1);
    assert_eq!(slot_bit(14), 0x4000);
    assert_eq!(slot_bit(63), 1 << 63);
    assert_eq!(slot_bit(64), 0);
}

/// One frame per tick: three reached frames take three ticks, each latching the oldest; the
/// notes of each tick come after its `VsyncTick` and its requests are the frame's own.
#[test]
fn one_latch_per_tick() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    for slot in [4, 2, 9] {
        queue(&shared, FIRST, frame(slot, None));
    }
    let (calls, presents) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }, latched(4)]);
    assert_eq!(presents, [request(4)]);
    let (calls, presents) = turn(&shared, tick_at(2));
    assert_eq!(
        notes(&calls),
        [Note::VsyncTick { tick: 2 }, latched(2), released(4)]
    );
    assert_eq!(presents, [request(2)]);
    let (calls, presents) = turn(&shared, tick_at(3));
    assert_eq!(
        notes(&calls),
        [Note::VsyncTick { tick: 3 }, latched(9), released(2)]
    );
    assert_eq!(presents, [request(9)]);
    let state = shared.lock();
    let client = state.presents().client(FIRST).unwrap();
    assert_eq!(client.queued().len(), 0);
    assert_eq!(client.shown().map(Frame::slot), Some(9));
}

/// A frame whose fence is not reached skips the tick, and the next tick after a drained value
/// reaches it latches it: not one value below the fence, exactly at it.
#[test]
fn an_unreached_fence_skips_a_tick() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST, frame(1, Some(5)));
    assert_eq!(shared.lock().watchers(fenced_on()), 1);
    let (calls, presents) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }]);
    assert_eq!(presents, []);
    drain_change(&shared, 4);
    let (calls, _) = turn(&shared, tick_at(2));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 2 }]);
    drain_change(&shared, 5);
    assert_eq!(shared.lock().watchers(fenced_on()), 0);
    assert!(!shared.syncpoints().is_watched(fenced_on()));
    let (calls, presents) = turn(&shared, tick_at(3));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 3 }, latched(1)]);
    assert_eq!(presents, [request(1)]);
}

/// The latch is FIFO: a later frame whose fence is reached does not pass an older one that is
/// not.
#[test]
fn a_reached_frame_does_not_pass_an_older_unreached_one() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST, frame(1, Some(5)));
    queue(&shared, FIRST, frame(2, None));
    let (calls, presents) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }]);
    assert_eq!(presents, []);
    drain_change(&shared, 5);
    let (calls, _) = turn(&shared, tick_at(2));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 2 }, latched(1)]);
    let (calls, _) = turn(&shared, tick_at(3));
    assert_eq!(
        notes(&calls),
        [Note::VsyncTick { tick: 3 }, latched(2), released(1)]
    );
}

/// A slot comes back exactly on the tick that latches the frame replacing it, and not before:
/// the shown frame keeps its slot while the frame behind it waits for its fence; the release
/// raises the client's notification and signals its event after the release's note.
#[test]
fn a_slot_returns_on_the_tick_that_latches_the_frame_replacing_it() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    queue(&shared, FIRST, frame(6, None));
    let (calls, _) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }, latched(6)]);
    assert!(!calls.contains(&Call::Signal(event(1))));
    queue(&shared, FIRST, frame(11, Some(3)));
    for n in 2..=3 {
        let (calls, _) = turn(&shared, tick_at(n));
        assert_eq!(notes(&calls), [Note::VsyncTick { tick: n }]);
        assert!(!calls.contains(&Call::Signal(event(1))));
        let mut state = shared.lock();
        assert!(state.presents().holds(FIRST, 6));
        assert_eq!(state.presents_mut().take_refresh(FIRST), Refresh::default());
    }
    drain_change(&shared, 3);
    let (calls, _) = turn(&shared, tick_at(4));
    let tail: Vec<Call> = calls[calls.len() - 3..].to_vec();
    assert_eq!(
        tail,
        [
            Call::Record(latched(11).event(GSV)),
            Call::Record(released(6).event(GSV)),
            Call::Signal(event(1)),
        ]
    );
    let mut state = shared.lock();
    assert!(!state.presents().holds(FIRST, 6));
    assert!(state.presents().holds(FIRST, 11));
    assert_eq!(
        state.presents_mut().take_refresh(FIRST),
        Refresh {
            released: slot_bit(6),
            raised: true
        }
    );
    // Taken once: the next refresh has nothing.
    assert_eq!(state.presents_mut().take_refresh(FIRST), Refresh::default());
}

/// A frame that does not hold its slot (a fence wait present) is released with its note, but
/// frees no slot, raises nothing and signals nothing; and a client with no event signals
/// nothing either but still raises.
#[test]
fn a_release_frees_a_slot_only_for_a_frame_that_holds_it() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    register(&shared, SECOND, SECOND_TOKEN, SECOND_BIT, None);
    let loose = NewFrame {
        holds_slot: false,
        ..frame(2, None)
    };
    queue(&shared, FIRST, loose);
    queue(&shared, FIRST, frame(3, None));
    queue(&shared, SECOND, frame(4, None));
    queue(&shared, SECOND, frame(5, None));
    assert!(!shared.lock().presents().holds(FIRST, 2));
    turn(&shared, tick_at(1));
    let (calls, _) = turn(&shared, tick_at(2));
    // Both clients in number order, each named; neither release signals.
    assert_eq!(
        notes(&calls),
        [
            Note::VsyncTick { tick: 2 },
            latched(3),
            released(2),
            latched_by(SECOND, 5),
            released_by(SECOND, 4)
        ]
    );
    assert!(!calls.iter().any(|call| matches!(call, Call::Signal(_))));
    let mut state = shared.lock();
    assert_eq!(state.presents_mut().take_refresh(FIRST), Refresh::default());
    assert_eq!(
        state.presents_mut().take_refresh(SECOND),
        Refresh {
            released: slot_bit(4),
            raised: true
        }
    );
}

/// A release whose event the kernel refuses is logged and the latch goes on to the next client.
#[test]
fn a_refused_release_signal_does_not_stop_the_latch() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    register(&shared, SECOND, SECOND_TOKEN, SECOND_BIT, Some(event(2)));
    for client in [FIRST, SECOND] {
        queue(&shared, client, frame(0, None));
        queue(&shared, client, frame(1, None));
    }
    turn(&shared, tick_at(1));
    let mut host = Journal {
        refused: BTreeSet::from([event(1)]),
        ..Journal::default()
    };
    shared.tick_vsync(&mut host, tick_at(2));
    assert!(host.calls.contains(&Call::Signal(event(1))));
    assert!(host.calls.contains(&Call::Signal(event(2))));
    let mut state = shared.lock();
    for client in [FIRST, SECOND] {
        assert_eq!(
            state.presents_mut().take_refresh(client).released,
            slot_bit(0)
        );
    }
}

/// A fence the counter already reaches when the frame is queued is not reached by that reading:
/// the frame keeps its watch, and the commit hands back a `SyncpointChanged` of the reading,
/// which the next drain records and which reaches it; one value short hands back nothing. The
/// comparison wraps: a counter 2^31 - 1 past the fence reaches it, 2^31 past does not.
#[test]
fn a_frame_is_reached_by_the_counter_it_is_queued_against() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    shared.syncpoints().increment_by(fenced_on(), 10);
    let cases = [
        (10, true),
        (11, false),
        (10_u32.wrapping_sub(i32::MAX.cast_unsigned()), true),
        (10_u32.wrapping_sub(1 << 31), false),
    ];
    for (slot, (value, reached)) in (0..).zip(cases) {
        let mut state = shared.lock();
        let before = state.watchers(fenced_on());
        let prepared = state
            .prepare_frame(FIRST, frame(slot, Some(value)))
            .unwrap();
        assert!(!prepared.frame.reached, "{value:#x}");
        assert_eq!(state.watchers(fenced_on()), before + 1, "{value:#x}");
        let change = state.commit_frame(prepared, Some(START), None);
        let expected = reached.then_some(GpuCompletion::SyncpointChanged {
            id: fenced_on(),
            value: 10,
        });
        assert_eq!(change, expected, "{value:#x}");
    }
    assert_eq!(shared.lock().presents().unreached_on(fenced_on()), 4);
    // The drained reading reaches the two it reaches, records the value it acted on, and gives
    // back their watches.
    shared
        .queue()
        .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
            id: fenced_on(),
            value: 10,
        }));
    let mut host = Journal::default();
    shared.drain(&mut host);
    assert_eq!(
        notes(&host.calls),
        [Note::SyncpointObserved {
            id: fenced_on(),
            value: 10
        }]
    );
    let state = shared.lock();
    assert_eq!(state.presents().unreached_on(fenced_on()), 2);
    assert_eq!(state.watchers(fenced_on()), 2);
    let reached: Vec<bool> = state
        .presents()
        .client(FIRST)
        .unwrap()
        .queued()
        .map(Frame::reached)
        .collect();
    assert_eq!(reached, [true, false, true, false]);
}

/// A drained value reaches a frame by the wrapping comparison too, and only frames on its own
/// syncpoint.
#[test]
fn a_drained_value_reaches_only_frames_on_its_syncpoint() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    let other = SyncpointId::new(0x18).unwrap();
    queue(&shared, FIRST, frame(0, Some(5)));
    let on_other = NewFrame {
        fence: Some(Fence {
            id: other,
            value: 5,
        }),
        ..frame(1, None)
    };
    queue(&shared, FIRST, on_other);
    let mut state = shared.lock();
    assert_eq!(state.presents_mut().reach(fenced_on(), 4), 0);
    assert_eq!(
        state
            .presents_mut()
            .reach(fenced_on(), 5 + i32::MAX.cast_unsigned()),
        1
    );
    assert_eq!(
        state.presents_mut().reach(fenced_on(), 5),
        0,
        "already reached"
    );
    assert_eq!(state.presents().unreached_on(other), 1);
    assert_eq!(
        state
            .presents_mut()
            .reach(other, 5u32.wrapping_add(1 << 31)),
        0
    );
    assert_eq!(state.presents_mut().reach(other, 6), 1);
}

/// The clock runs while a present waits: a frame queued with no listener starts it one interval
/// after the time given, the tick that latches the last frame stops it, and a listener leaving
/// while a frame waits does not stop it.
#[test]
fn the_clock_runs_while_a_present_waits() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    assert_eq!(shared.vsync_deadline(), None);
    queue(&shared, FIRST, frame(0, Some(1)));
    assert_eq!(shared.vsync_deadline(), Some(tick_at(1)));
    // A listener joins and leaves: the waiting frame keeps the clock running.
    assert_eq!(shared.lock().listen_vsync(event(7), START), Ok(true));
    assert!(shared.lock().unlisten_vsync(event(7)));
    assert_eq!(shared.vsync_deadline(), Some(tick_at(1)));
    turn(&shared, tick_at(1));
    assert_eq!(shared.vsync_deadline(), Some(tick_at(2)));
    drain_change(&shared, 1);
    let (calls, _) = turn(&shared, tick_at(2));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 2 }, latched(0)]);
    assert_eq!(shared.vsync_deadline(), None);
    // A running clock keeps its deadline when a frame is queued.
    assert_eq!(shared.lock().listen_vsync(event(7), HostNanos(9)), Ok(true));
    let mut state = shared.lock();
    let prepared = state.prepare_frame(FIRST, frame(1, None)).unwrap();
    assert_eq!(
        state.commit_frame(prepared, Some(HostNanos(1_000)), None),
        None
    );
    drop(state);
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(9 + INTERVAL_NS)));
}

/// 201's table admits exactly [`MAX_PRESENT_CLIENTS`] clients, each under the lowest number no
/// client holds; one token may be registered many times (each a client of its own session);
/// a full table is refused and nothing changes; a removed client's number is given again.
#[test]
fn the_table_takes_clients_up_to_its_bound() {
    let shared = fresh();
    let mut numbers = Vec::new();
    for bit in 0..u32::try_from(MAX_PRESENT_CLIENTS).unwrap() {
        let number = shared
            .lock()
            .presents_mut()
            .register(FIRST_TOKEN, bit % CLIENT_NOTIFICATIONS, None)
            .unwrap();
        numbers.push(usize::from(number));
    }
    assert_eq!(numbers, (0..MAX_PRESENT_CLIENTS).collect::<Vec<_>>());
    let refused = shared.lock().presents_mut().register(SECOND_TOKEN, 0, None);
    assert_eq!(refused, Err(RegisterError::Full));
    assert_eq!(
        RegisterError::Full.to_string(),
        "the present table already holds the most clients it takes (128)"
    );
    let state = shared.lock();
    assert_eq!(state.presents().clients().count(), MAX_PRESENT_CLIENTS);
    assert!(
        state
            .presents()
            .clients()
            .all(|(_, client)| client.token() == FIRST_TOKEN)
    );
    drop(state);
    // Freeing two numbers: the lower is given first, then the higher.
    assert!(shared.lock().unregister_present(9));
    assert!(shared.lock().unregister_present(4));
    let mut state = shared.lock();
    let table = state.presents_mut();
    assert_eq!(table.register(SECOND_TOKEN, 1, None), Ok(4));
    assert_eq!(table.register(SECOND_TOKEN, 2, None), Ok(9));
    assert_eq!(
        table.register(SECOND_TOKEN, 3, None),
        Err(RegisterError::Full)
    );
    assert_eq!(
        table.client(4).map(PresentClient::token),
        Some(SECOND_TOKEN)
    );
    assert_eq!(table.client(4).map(PresentClient::available_bit), Some(1));
}

/// Queuing refuses a token with no entry, then a client with [`SLOTS`] frames queued (the
/// fifteenth is taken), then a fence it cannot watch; nothing changes on a refusal.
#[test]
fn queuing_refuses_in_order() {
    let shared = fresh();
    assert_eq!(
        shared.lock().prepare_frame(FIRST, frame(0, Some(1))),
        Err(QueueError::NoClient { client: FIRST })
    );
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    shared.set_watchers(fenced_on(), u32::MAX);
    let error = shared
        .lock()
        .prepare_frame(FIRST, frame(0, Some(1)))
        .unwrap_err();
    assert_eq!(
        error,
        QueueError::Watch(WatchError::TooManyWaiters(fenced_on()))
    );
    assert_eq!(shared.lock().watchers(fenced_on()), u32::MAX);
    shared.set_watchers(fenced_on(), 0);
    for slot in 0..SLOTS {
        let loose = NewFrame {
            holds_slot: false,
            ..frame(slot, None)
        };
        queue(&shared, FIRST, loose);
    }
    // Full before the watch: the counter's watch is not taken.
    let error = shared
        .lock()
        .prepare_frame(FIRST, frame(0, Some(1)))
        .unwrap_err();
    assert_eq!(error, QueueError::Full { client: FIRST });
    assert_eq!(shared.lock().watchers(fenced_on()), 0);
    assert_eq!(
        error.to_string(),
        "present table client 0 already has the most frames queued (15)"
    );
    assert_eq!(
        QueueError::NoClient { client: 7 }.to_string(),
        "the present table has no client 7"
    );
}

/// An abandoned frame gives back its watch, also one whose fence the counter already reached
/// (its change is not pushed); an unfenced one has none to give.
#[test]
fn an_abandoned_frame_gives_back_its_watch() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    let mut state = shared.lock();
    let prepared = state.prepare_frame(FIRST, frame(0, Some(1))).unwrap();
    assert_eq!(state.watchers(fenced_on()), 1);
    state.abandon_frame(prepared);
    assert_eq!(state.watchers(fenced_on()), 0);
    drop(state);
    shared.syncpoints().increment_by(fenced_on(), 1);
    let mut state = shared.lock();
    let prepared = state.prepare_frame(FIRST, frame(0, Some(1))).unwrap();
    assert_eq!(state.watchers(fenced_on()), 1);
    state.abandon_frame(prepared);
    assert_eq!(state.watchers(fenced_on()), 0);
    let prepared = state.prepare_frame(FIRST, frame(0, None)).unwrap();
    state.abandon_frame(prepared);
    assert_eq!(state.watchers(fenced_on()), 0);
    assert!(!state.presents().has_work());
    assert!(shared.queue().is_empty());
    // The next attempt succeeds.
    let prepared = state.prepare_frame(FIRST, frame(0, Some(2))).unwrap();
    assert_eq!(state.commit_frame(prepared, Some(START), None), None);
    assert_eq!(state.watchers(fenced_on()), 1);
}

/// 202's removal drops a client's queued and shown frames and gives back each unreached
/// frame's watch; another client is untouched.
#[test]
fn unregistering_gives_back_the_watches() {
    let shared = fresh();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    register(&shared, SECOND, SECOND_TOKEN, SECOND_BIT, None);
    queue(&shared, FIRST, frame(0, None));
    queue(&shared, SECOND, frame(0, Some(9)));
    turn(&shared, tick_at(1));
    queue(&shared, FIRST, frame(1, Some(9)));
    queue(&shared, FIRST, frame(2, Some(9)));
    assert_eq!(shared.lock().watchers(fenced_on()), 3);
    assert!(shared.lock().unregister_present(FIRST));
    assert!(!shared.lock().unregister_present(FIRST));
    let state = shared.lock();
    assert_eq!(state.watchers(fenced_on()), 1);
    assert!(state.presents().client(FIRST).is_none());
    assert_eq!(state.presents().client(SECOND).unwrap().queued().len(), 1);
}

/// 11's event reaches the client's entry; a number the table does not have changes nothing.
#[test]
fn the_notification_event_reaches_the_entry() {
    let mut table = Presents::new();
    table.register(FIRST_TOKEN, FIRST_BIT, None).unwrap();
    table.set_event(FIRST, event(4));
    table.set_event(SECOND, event(5));
    assert_eq!(table.client(FIRST).unwrap().event(), Some(event(4)));
    assert!(table.client(SECOND).is_none());
    assert_eq!(table.take_refresh(SECOND), Refresh::default());
}

// ----- The snapshot -------------------------------------------------------------------------

/// A vi2 server with no request pending, for the snapshot calls.
fn no_server(shared: &Arc<Shared>) -> Server {
    vi2_server(UnknownPolicy::Strict, Arc::clone(shared))
}

fn shared_arc() -> Arc<Shared> {
    Arc::new(fresh())
}

/// A saved frame of `slot` with every field distinct.
fn saved_frame(slot: u8, fence: Option<(u8, u32)>, reached: bool) -> SavedFrame {
    SavedFrame {
        slot,
        image: 0x0A0B_0C0D,
        address: 0x0102_0304_0506,
        width: 0x11,
        height: 0x12,
        pitch: 0x13,
        block_height_log2: 0x14,
        format: 0x15,
        shown: [0x16, 0x17, 0x18, 0x19],
        layer: 0x1A,
        display: 0x1B,
        fence,
        reached,
        holds_slot: true,
    }
}

/// A saved client of number `number`, its token derived from it.
fn saved_client(number: u8) -> SavedClient {
    SavedClient {
        number,
        token: 0x50 + u64::from(number),
        available_bit: 2,
        event: Some(ObjectId::from_parts(3, 4)),
        queued: Vec::new(),
        shown: None,
        released: 0,
        raised: false,
        waiting: None,
    }
}

/// The bytes of a snapshot of `clients`.
fn snapshot(clients: &[SavedClient]) -> Vec<u8> {
    let mut bytes = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    bytes.extend(postcard::to_allocvec(clients).unwrap());
    bytes
}

#[test]
fn the_snapshot_layout_is_pinned() {
    // One client: number 0x0E, token 0x21, bit 5, event (6, 7); one frame queued (slot 2,
    // image 0x31, every request field distinct, fenced on syncpoint 0x17 at 0x41, not reached,
    // holds its slot); the shown frame (slot 3, no fence, reached); released slot 1; raised; no
    // held reply.
    let shared = shared_arc();
    let number = 0x0E;
    for earlier in 0..number {
        register(&shared, earlier, 0x99, 0, None);
    }
    register(&shared, number, 0x21, 5, Some(ObjectId::from_parts(6, 7)));
    for earlier in 0..number {
        assert!(shared.lock().unregister_present(earlier));
    }
    let shown = NewFrame {
        slot: 3,
        image: 0x32,
        request: request(1),
        fence: None,
        holds_slot: true,
    };
    let behind = NewFrame {
        slot: 2,
        image: 0x31,
        request: request(2),
        fence: Some(Fence {
            id: fenced_on(),
            value: 0x41,
        }),
        holds_slot: true,
    };
    queue(&shared, number, NewFrame { slot: 1, ..shown });
    turn(&shared, tick_at(1));
    queue(&shared, number, shown);
    turn(&shared, tick_at(2));
    queue(&shared, number, behind);
    let mut expected = vec![b'S', b'Q', b'P', b'R', 0x01];
    expected.extend([0x01, 0x0E, 0x21, 0x05, 0x01, 0x06, 0x07]);
    // The queued frame: slot, image, address (0x200000 as a varint), width 102, height 202,
    // pitch 302, block height 7, format 1, shown 0, 0, 12, 22, layer 42, display 52, the fence
    // (some: 0x17, 0x41), not reached, holds.
    expected.extend([0x01, 0x02, 0x31, 0x80, 0x80, 0x80, 0x01]);
    expected.extend([
        0x66, 0xCA, 0x01, 0xAE, 0x02, 0x07, 0x01, 0x00, 0x00, 0x0C, 0x16,
    ]);
    expected.extend([0x2A, 0x34, 0x01, 0x17, 0x41, 0x00, 0x01]);
    // The shown frame: some; slot 3, image 0x32, address 0x100000, width 101, height 201,
    // pitch 301, 7, 1, shown 0, 0, 11, 21, layer 41, display 51, no fence, reached, holds.
    expected.extend([0x01, 0x03, 0x32, 0x80, 0x80, 0x40]);
    expected.extend([
        0x65, 0xC9, 0x01, 0xAD, 0x02, 0x07, 0x01, 0x00, 0x00, 0x0B, 0x15,
    ]);
    expected.extend([0x29, 0x33, 0x00, 0x01, 0x01]);
    // Released slot 1 (bit 2), raised, no held reply.
    expected.extend([0x02, 0x01, 0x00]);
    let server = no_server(&shared);
    assert_eq!(shared.save_presents(&server).unwrap(), expected);
}

/// The state live operations leave restores, and the restored table is used: its waiting frame
/// latches on the next tick once its fence is drained.
#[test]
fn a_restored_table_is_used_as_saved() {
    let source = shared_arc();
    register(&source, FIRST, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    queue(&source, FIRST, frame(0, None));
    turn(&source, tick_at(1));
    queue(&source, FIRST, frame(1, Some(4)));
    let bytes = source.save_presents(&no_server(&source)).unwrap();

    let target = shared_arc();
    target.lock().watch(fenced_on()).unwrap();
    target
        .restore_presents(&bytes, &no_server(&target))
        .unwrap();
    assert_eq!(target.lock().presents(), source.lock().presents());
    assert!(target.lock().listen_vsync(event(9), START).is_ok());
    drain_change(&target, 4);
    let (calls, presents) = turn(&target, tick_at(1));
    assert_eq!(
        notes(&calls),
        [Note::VsyncTick { tick: 1 }, latched(1), released(0)]
    );
    assert_eq!(presents, [request(1)]);
}

/// A forged snapshot of one client, number 5, changed by `change`.
fn one_client(change: impl Fn(&mut SavedClient)) -> Vec<SavedClient> {
    let mut client = saved_client(5);
    change(&mut client);
    vec![client]
}

/// The client checks' faults, each with its error.
fn client_faults() -> Vec<(Vec<SavedClient>, PresentSnapshotError)> {
    vec![
        (
            (0..=last_number() + 1).map(saved_client).collect(),
            PresentSnapshotError::TooManyClients {
                count: MAX_PRESENT_CLIENTS + 1,
            },
        ),
        (
            vec![saved_client(1), saved_client(last_number() + 1)],
            PresentSnapshotError::ClientNumber {
                index: 1,
                number: last_number() + 1,
            },
        ),
        (
            vec![saved_client(2), saved_client(2)],
            PresentSnapshotError::NumbersNotAscending { index: 1 },
        ),
        (
            vec![saved_client(3), saved_client(2)],
            PresentSnapshotError::NumbersNotAscending { index: 1 },
        ),
        (
            vec![saved_client(1), with_token(saved_client(2), 0)],
            PresentSnapshotError::ZeroToken { index: 1 },
        ),
        // Two faults at once: the earlier check is answered.
        (
            vec![with_token(saved_client(last_number() + 1), 0)],
            PresentSnapshotError::ClientNumber {
                index: 0,
                number: last_number() + 1,
            },
        ),
        (
            vec![saved_client(3), with_token(saved_client(3), 0)],
            PresentSnapshotError::NumbersNotAscending { index: 1 },
        ),
        (
            one_client(|c| c.available_bit = CLIENT_NOTIFICATIONS),
            PresentSnapshotError::NotificationBit { index: 0 },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(0, None, true)];
                c.waiting = Some(SavedWaiting {
                    token: None,
                    ahead: 1,
                });
            }),
            PresentSnapshotError::Ahead {
                index: 0,
                ahead: 1,
                queued: 1,
            },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(0, None, true)];
                c.waiting = Some(SavedWaiting {
                    token: Some(postcard::from_bytes(&[3]).unwrap()),
                    ahead: 0,
                });
            }),
            PresentSnapshotError::NotPending { index: 0, token: 3 },
        ),
    ]
}

/// The released-slot checks' faults, each with its error.
fn release_faults() -> Vec<(Vec<SavedClient>, PresentSnapshotError)> {
    vec![
        (
            one_client(|c| {
                c.released = ALL_SLOTS + 1;
                c.raised = true;
            }),
            PresentSnapshotError::Released { index: 0 },
        ),
        // A queued frame and a shown one each holding a released slot, then both, each way
        // round: the lowest slot is named.
        (
            one_client(|c| {
                c.queued = vec![saved_frame(0, None, true)];
                c.released = slot_bit(0);
                c.raised = true;
            }),
            PresentSnapshotError::ReleasedHeld { index: 0, slot: 0 },
        ),
        (
            one_client(|c| {
                c.shown = Some(saved_frame(4, None, true));
                c.released = slot_bit(4);
                c.raised = true;
            }),
            PresentSnapshotError::ReleasedHeld { index: 0, slot: 4 },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(6, None, true)];
                c.shown = Some(saved_frame(4, None, true));
                c.released = slot_bit(4) | slot_bit(6);
                c.raised = true;
            }),
            PresentSnapshotError::ReleasedHeld { index: 0, slot: 4 },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(4, None, true)];
                c.shown = Some(saved_frame(6, None, true));
                c.released = slot_bit(4) | slot_bit(6);
                c.raised = true;
            }),
            PresentSnapshotError::ReleasedHeld { index: 0, slot: 4 },
        ),
        (
            one_client(|c| c.raised = true),
            PresentSnapshotError::RaisedWithoutRelease { index: 0 },
        ),
        (
            one_client(|c| c.released = slot_bit(1)),
            PresentSnapshotError::ReleasedNotRaised { index: 0 },
        ),
        // Two faults at once: the range before the held slots, the held slots before the
        // notification.
        (
            one_client(|c| c.released = slot_bit(SLOTS)),
            PresentSnapshotError::Released { index: 0 },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(0, None, true)];
                c.released = slot_bit(0);
            }),
            PresentSnapshotError::ReleasedHeld { index: 0, slot: 0 },
        ),
    ]
}

/// The last client number the table gives.
fn last_number() -> u8 {
    u8::try_from(MAX_PRESENT_CLIENTS - 1).unwrap()
}

/// `client` with the token `token`.
fn with_token(client: SavedClient, token: u64) -> SavedClient {
    SavedClient { token, ..client }
}

/// The frame checks' faults, each with its error.
fn frame_faults() -> Vec<(Vec<SavedClient>, PresentSnapshotError)> {
    vec![
        (
            one_client(|c| {
                c.queued = (0..=SLOTS)
                    .map(|slot| SavedFrame {
                        holds_slot: false,
                        ..saved_frame(slot % SLOTS, None, true)
                    })
                    .collect();
            }),
            PresentSnapshotError::TooManyFrames {
                index: 0,
                count: usize::from(SLOTS) + 1,
            },
        ),
        (
            one_client(|c| c.shown = Some(saved_frame(SLOTS, None, true))),
            PresentSnapshotError::Slot {
                index: 0,
                slot: SLOTS,
            },
        ),
        (
            one_client(|c| c.queued = vec![saved_frame(0, Some((192, 1)), false)]),
            PresentSnapshotError::FenceSyncpoint { index: 0, id: 192 },
        ),
        (
            one_client(|c| c.queued = vec![saved_frame(0, None, false)]),
            PresentSnapshotError::UnfencedUnreached { index: 0 },
        ),
        (
            one_client(|c| c.shown = Some(saved_frame(0, Some((0x17, 1)), false))),
            PresentSnapshotError::ShownUnreached { index: 0 },
        ),
        (
            one_client(|c| {
                c.queued = vec![saved_frame(4, None, true)];
                c.shown = Some(saved_frame(4, None, true));
            }),
            PresentSnapshotError::SlotHeldTwice { index: 0, slot: 4 },
        ),
        (
            one_client(|c| c.queued = vec![saved_frame(0, Some((0x17, 1)), false)]),
            PresentSnapshotError::FramesOutnumberWatchers {
                id: 0x17,
                frames: 1,
                waits: 0,
                waiters: 0,
            },
        ),
    ]
}

/// Each restore check refuses its own fault with its exact error, and changes nothing.
#[test]
fn each_restore_check_refuses_its_fault() {
    let faults = [client_faults(), release_faults(), frame_faults()].concat();
    for (clients, expected) in faults {
        let shared = shared_arc();
        register(&shared, FIRST, 0x99, 1, None);
        let before = shared.save_presents(&no_server(&shared)).unwrap();
        let error = shared
            .restore_presents(&snapshot(&clients), &no_server(&shared))
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(shared.save_presents(&no_server(&shared)).unwrap(), before);
    }
}

/// The edges of the restore checks, each accepted exactly at its limit: [`MAX_PRESENT_CLIENTS`]
/// clients, the last client number, [`SLOTS`] frames, the last slot, the last syncpoint, the
/// last notification bit, every slot released (none held: the frames do not hold theirs), a
/// reply behind one frame fewer than are queued, and an unreached frame with exactly the
/// waiters it needs.
#[test]
fn the_restore_checks_accept_their_edges() {
    let shared = shared_arc();
    let last_id = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    let last = SyncpointId::new(last_id).unwrap();
    shared.lock().watch(last).unwrap();
    let mut edge = saved_client(0);
    edge.available_bit = CLIENT_NOTIFICATIONS - 1;
    edge.queued = (0..SLOTS)
        .map(|slot| SavedFrame {
            holds_slot: false,
            ..saved_frame(slot, None, true)
        })
        .collect();
    edge.queued[0] = SavedFrame {
        holds_slot: false,
        ..saved_frame(SLOTS - 1, Some((last_id, 3)), false)
    };
    edge.released = ALL_SLOTS;
    edge.raised = true;
    edge.waiting = Some(SavedWaiting {
        token: None,
        ahead: u32::from(SLOTS) - 1,
    });
    let mut clients = vec![edge];
    clients.extend((1..=last_number()).map(saved_client));
    shared
        .restore_presents(&snapshot(&clients), &no_server(&shared))
        .unwrap();
    assert_eq!(
        shared.lock().presents().clients().count(),
        MAX_PRESENT_CLIENTS
    );
    assert_eq!(shared.lock().presents().unreached_on(last), 1);
    assert!(shared.lock().presents().client(last_number()).is_some());
}

/// A client's released slots and the slots its frames hold are disjoint: every other slot
/// released beside a held one is taken, and the held one released too is refused.
#[test]
fn the_released_slots_and_the_held_ones_are_disjoint() {
    let shared = shared_arc();
    let held = |released| {
        one_client(|c| {
            c.queued = vec![saved_frame(SLOTS - 1, None, true)];
            c.shown = Some(saved_frame(0, None, true));
            c.released = released;
            c.raised = true;
        })
    };
    let free = ALL_SLOTS & !slot_bit(0) & !slot_bit(SLOTS - 1);
    let server = no_server(&shared);
    assert_eq!(
        shared.restore_presents(&snapshot(&held(free)), &server),
        Ok(())
    );
    let error = shared
        .restore_presents(&snapshot(&held(free | slot_bit(SLOTS - 1))), &server)
        .unwrap_err();
    assert_eq!(
        error,
        PresentSnapshotError::ReleasedHeld {
            index: 0,
            slot: SLOTS - 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the present table snapshot's client 0 has slot 14 released while a frame holds it"
    );
    for (error, text) in [
        (
            PresentSnapshotError::RaisedWithoutRelease { index: 0 },
            "the present table snapshot's client 0 has its notification raised with no slot released",
        ),
        (
            PresentSnapshotError::ReleasedNotRaised { index: 0 },
            "the present table snapshot's client 0 has a slot released with its notification not raised",
        ),
        (
            PresentSnapshotError::ClientNumber {
                index: 0,
                number: 128,
            },
            "the present table snapshot's client 0 has the number 128, not below 128",
        ),
    ] {
        assert_eq!(error.to_string(), text);
    }
}

/// A frame's fence on a syncpoint freed after 229 queued it is the state a live table reaches
/// (a channel closed while the present waits): it saves, restores, and latches once a drained
/// value reaches it, as it does live.
#[test]
fn a_restore_accepts_a_frame_fenced_on_a_freed_syncpoint() {
    let shared = shared_arc();
    let id = shared.syncpoints().allocate().unwrap();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    let fenced = NewFrame {
        fence: Some(Fence { id, value: 2 }),
        ..frame(3, None)
    };
    queue(&shared, FIRST, fenced);
    shared.syncpoints().free(id).unwrap();
    assert!(!shared.syncpoints().is_allocated(id));
    let bytes = shared.save_presents(&no_server(&shared)).unwrap();
    shared
        .restore_presents(&bytes, &no_server(&shared))
        .unwrap();
    shared
        .queue()
        .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
            id,
            value: 2,
        }));
    shared.drain(&mut Journal::default());
    let (calls, presents) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }, latched(3)]);
    assert_eq!(presents, [request(3)]);
}

/// The watch check counts armed syncpt waits beside the frames: one waiter for a wait and a
/// frame is refused, two are taken.
#[test]
fn the_watch_check_counts_armed_waits_too() {
    let shared = shared_arc();
    let wait = Wait::new(fenced_on(), 100, CompareMode::Wrapping32).unwrap();
    shared.lock().arm(event(1), wait).unwrap();
    let mut client = saved_client(1);
    client.queued = vec![saved_frame(0, Some((0x17, 1)), false)];
    let bytes = snapshot(&[client]);
    assert_eq!(
        shared.restore_presents(&bytes, &no_server(&shared)),
        Err(PresentSnapshotError::FramesOutnumberWatchers {
            id: 0x17,
            frames: 1,
            waits: 1,
            waiters: 1
        })
    );
    shared.lock().watch(fenced_on()).unwrap();
    assert_eq!(shared.restore_presents(&bytes, &no_server(&shared)), Ok(()));
}

/// The header is checked before the body: a bad magic, another version, trailing bytes.
#[test]
fn the_header_is_checked_before_the_body() {
    let shared = shared_arc();
    let server = no_server(&shared);
    let good = snapshot(&[]);
    let mut magic = good.clone();
    magic[0] = b'X';
    assert_eq!(
        shared.restore_presents(&magic, &server),
        Err(PresentSnapshotError::BadMagic)
    );
    let mut version = good.clone();
    version[4] = 2;
    assert_eq!(
        shared.restore_presents(&version, &server),
        Err(PresentSnapshotError::UnsupportedVersion { found: 2 })
    );
    let mut trailing = good;
    trailing.push(0);
    let error = shared.restore_presents(&trailing, &server).unwrap_err();
    assert_eq!(error, PresentSnapshotError::TrailingBytes { count: 1 });
    assert_eq!(
        error.to_string(),
        "1 unexpected bytes after the present table snapshot"
    );
}

/// Saving runs the restore's relation checks: a frame whose watch was taken behind the table's
/// back is refused, and so are a released slot a frame holds and a notification raised with no
/// slot released, each corrupted in the live table.
#[test]
fn saving_refuses_what_restoring_would() {
    let shared = shared_arc();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST, frame(0, Some(1)));
    shared.set_watchers(fenced_on(), 0);
    assert_eq!(
        shared.save_presents(&no_server(&shared)),
        Err(PresentSnapshotError::FramesOutnumberWatchers {
            id: 0x17,
            frames: 1,
            waits: 0,
            waiters: 0
        })
    );
    shared.set_watchers(fenced_on(), 1);
    assert!(shared.save_presents(&no_server(&shared)).is_ok());
    let corrupt = |released: u64, raised: bool| {
        let mut state = shared.lock();
        let client = state.presents_mut().clients.get_mut(&FIRST).unwrap();
        client.released = released;
        client.raised = raised;
    };
    corrupt(slot_bit(0), true);
    assert_eq!(
        shared.save_presents(&no_server(&shared)),
        Err(PresentSnapshotError::ReleasedHeld { index: 0, slot: 0 })
    );
    corrupt(0, true);
    assert_eq!(
        shared.save_presents(&no_server(&shared)),
        Err(PresentSnapshotError::RaisedWithoutRelease { index: 0 })
    );
    corrupt(slot_bit(1), false);
    assert_eq!(
        shared.save_presents(&no_server(&shared)),
        Err(PresentSnapshotError::ReleasedNotRaised { index: 0 })
    );
}

/// The vsync clock's snapshot keeps a deadline with no listener only while the table has work,
/// on save and on restore.
#[test]
fn the_clock_s_snapshot_takes_a_deadline_with_no_listener_only_while_a_present_waits() {
    let shared = shared_arc();
    register(&shared, FIRST, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST, frame(0, Some(1)));
    let bytes = shared.save_vsync().unwrap();
    shared.restore_vsync(&bytes).unwrap();
    let idle = shared_arc();
    assert_eq!(
        idle.restore_vsync(&bytes),
        Err(VsyncSnapshotError::DeadlineWithoutListeners)
    );
    assert_eq!(
        VsyncSnapshotError::DeadlineWithoutListeners.to_string(),
        "the vsync clock snapshot has a deadline but no listeners, and no present waits for a tick"
    );
}
