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

/// The two clients, the second's token above the first's, and their bits.
const FIRST_TOKEN: u64 = 0x1111_0000_0000_0001;
const SECOND_TOKEN: u64 = 0x2222_0000_0000_0002;
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

/// Registers `token` with bit `bit` and event `notify`.
fn register(shared: &Shared, token: u64, bit: u32, notify: Option<ObjectId>) {
    shared
        .lock()
        .presents_mut()
        .register(token, bit, notify)
        .unwrap();
}

/// Queues `new` for `token`, starting a stopped clock at [`START`].
fn queue(shared: &Shared, token: u64, new: NewFrame) {
    let mut state = shared.lock();
    let prepared = state.prepare_frame(token, new).unwrap();
    state.commit_frame(prepared, Some(START), None);
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

fn latched(slot: u8) -> Note {
    Note::FrameLatched {
        slot,
        image: 0x100 + u32::from(slot),
    }
}

fn released(slot: u8) -> Note {
    Note::BufferReleased {
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
    register(&shared, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    for slot in [4, 2, 9] {
        queue(&shared, FIRST_TOKEN, frame(slot, None));
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
    let client = state.presents().client(FIRST_TOKEN).unwrap();
    assert_eq!(client.queued().len(), 0);
    assert_eq!(client.shown().map(Frame::slot), Some(9));
}

/// A frame whose fence is not reached skips the tick, and the next tick after a drained value
/// reaches it latches it: not one value below the fence, exactly at it.
#[test]
fn an_unreached_fence_skips_a_tick() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST_TOKEN, frame(1, Some(5)));
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
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST_TOKEN, frame(1, Some(5)));
    queue(&shared, FIRST_TOKEN, frame(2, None));
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
    register(&shared, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    queue(&shared, FIRST_TOKEN, frame(6, None));
    let (calls, _) = turn(&shared, tick_at(1));
    assert_eq!(notes(&calls), [Note::VsyncTick { tick: 1 }, latched(6)]);
    assert!(!calls.contains(&Call::Signal(event(1))));
    queue(&shared, FIRST_TOKEN, frame(11, Some(3)));
    for n in 2..=3 {
        let (calls, _) = turn(&shared, tick_at(n));
        assert_eq!(notes(&calls), [Note::VsyncTick { tick: n }]);
        assert!(!calls.contains(&Call::Signal(event(1))));
        let mut state = shared.lock();
        assert!(state.presents().holds(FIRST_TOKEN, 6));
        assert_eq!(
            state.presents_mut().take_refresh(FIRST_TOKEN),
            Refresh::default()
        );
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
    assert!(!state.presents().holds(FIRST_TOKEN, 6));
    assert!(state.presents().holds(FIRST_TOKEN, 11));
    assert_eq!(
        state.presents_mut().take_refresh(FIRST_TOKEN),
        Refresh {
            released: slot_bit(6),
            raised: true
        }
    );
    // Taken once: the next refresh has nothing.
    assert_eq!(
        state.presents_mut().take_refresh(FIRST_TOKEN),
        Refresh::default()
    );
}

/// A frame that does not hold its slot (a fence wait present) is released with its note, but
/// frees no slot, raises nothing and signals nothing; and a client with no event signals
/// nothing either but still raises.
#[test]
fn a_release_frees_a_slot_only_for_a_frame_that_holds_it() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    register(&shared, SECOND_TOKEN, SECOND_BIT, None);
    let loose = NewFrame {
        holds_slot: false,
        ..frame(2, None)
    };
    queue(&shared, FIRST_TOKEN, loose);
    queue(&shared, FIRST_TOKEN, frame(3, None));
    queue(&shared, SECOND_TOKEN, frame(4, None));
    queue(&shared, SECOND_TOKEN, frame(5, None));
    assert!(!shared.lock().presents().holds(FIRST_TOKEN, 2));
    turn(&shared, tick_at(1));
    let (calls, _) = turn(&shared, tick_at(2));
    // Both clients in token order; neither release signals.
    assert_eq!(
        notes(&calls),
        [
            Note::VsyncTick { tick: 2 },
            latched(3),
            released(2),
            latched(5),
            released(4)
        ]
    );
    assert!(!calls.iter().any(|call| matches!(call, Call::Signal(_))));
    let mut state = shared.lock();
    assert_eq!(
        state.presents_mut().take_refresh(FIRST_TOKEN),
        Refresh::default()
    );
    assert_eq!(
        state.presents_mut().take_refresh(SECOND_TOKEN),
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
    register(&shared, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    register(&shared, SECOND_TOKEN, SECOND_BIT, Some(event(2)));
    for token in [FIRST_TOKEN, SECOND_TOKEN] {
        queue(&shared, token, frame(0, None));
        queue(&shared, token, frame(1, None));
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
    for token in [FIRST_TOKEN, SECOND_TOKEN] {
        assert_eq!(state.presents_mut().take_refresh(token).released, slot_bit(0));
    }
}

/// A fence the counter already reaches when the frame is queued is reached at once and holds no
/// watch; one value short holds one. The comparison wraps: a counter 2^31 - 1 past the fence
/// reaches it, 2^31 past does not.
#[test]
fn a_frame_is_reached_by_the_counter_it_is_queued_against() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
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
        let prepared = state.prepare_frame(FIRST_TOKEN, frame(slot, Some(value))).unwrap();
        assert_eq!(prepared.frame.reached, reached, "{value:#x}");
        assert_eq!(
            state.watchers(fenced_on()),
            before + u32::from(!reached),
            "{value:#x}"
        );
        state.commit_frame(prepared, Some(START), None);
    }
    assert_eq!(shared.lock().presents().unreached_on(fenced_on()), 2);
}

/// A drained value reaches a frame by the wrapping comparison too, and only frames on its own
/// syncpoint.
#[test]
fn a_drained_value_reaches_only_frames_on_its_syncpoint() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    let other = SyncpointId::new(0x18).unwrap();
    queue(&shared, FIRST_TOKEN, frame(0, Some(5)));
    let on_other = NewFrame {
        fence: Some(Fence { id: other, value: 5 }),
        ..frame(1, None)
    };
    queue(&shared, FIRST_TOKEN, on_other);
    let mut state = shared.lock();
    assert_eq!(state.presents_mut().reach(fenced_on(), 4), 0);
    assert_eq!(state.presents_mut().reach(fenced_on(), 5 + i32::MAX.cast_unsigned()), 1);
    assert_eq!(state.presents_mut().reach(fenced_on(), 5), 0, "already reached");
    assert_eq!(state.presents().unreached_on(other), 1);
    assert_eq!(state.presents_mut().reach(other, 5u32.wrapping_add(1 << 31)), 0);
    assert_eq!(state.presents_mut().reach(other, 6), 1);
}

/// The clock runs while a present waits: a frame queued with no listener starts it one interval
/// after the time given, the tick that latches the last frame stops it, and a listener leaving
/// while a frame waits does not stop it.
#[test]
fn the_clock_runs_while_a_present_waits() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    assert_eq!(shared.vsync_deadline(), None);
    queue(&shared, FIRST_TOKEN, frame(0, Some(1)));
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
    let prepared = state.prepare_frame(FIRST_TOKEN, frame(1, None)).unwrap();
    state.commit_frame(prepared, Some(HostNanos(1_000)), None);
    drop(state);
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(9 + INTERVAL_NS)));
}

/// 201's table admits exactly [`MAX_PRESENT_CLIENTS`] clients; it refuses a token it holds
/// before a full table, and nothing changes on a refusal.
#[test]
fn the_table_takes_clients_up_to_its_bound() {
    let mut table = Presents::new();
    let tokens = 1..=u64::try_from(MAX_PRESENT_CLIENTS).unwrap();
    for token in tokens.clone() {
        assert_eq!(table.register(token, 0, None), Ok(()));
    }
    let next = u64::try_from(MAX_PRESENT_CLIENTS).unwrap() + 1;
    assert_eq!(table.register(next, 0, None), Err(RegisterError::Full));
    assert_eq!(
        table.register(1, 0, None),
        Err(RegisterError::Held { token: 1 })
    );
    assert_eq!(table.clients().count(), MAX_PRESENT_CLIENTS);
    assert_eq!(
        RegisterError::Full.to_string(),
        "the present table already holds the most clients it takes (128)"
    );
    assert_eq!(
        RegisterError::Held { token: 0x2A }.to_string(),
        "another vi2 session's client already has the token 0x2a"
    );
    // Below the bound, a held token is refused too.
    let mut table = Presents::new();
    table.register(7, 1, None).unwrap();
    assert_eq!(
        table.register(7, 2, None),
        Err(RegisterError::Held { token: 7 })
    );
    assert_eq!(table.client(7).unwrap().available_bit(), 1);
}

/// Queuing refuses a token with no entry, then a client with [`SLOTS`] frames queued (the
/// fifteenth is taken), then a fence it cannot watch; nothing changes on a refusal.
#[test]
fn queuing_refuses_in_order() {
    let shared = fresh();
    assert_eq!(
        shared.lock().prepare_frame(FIRST_TOKEN, frame(0, Some(1))),
        Err(QueueError::NoClient { token: FIRST_TOKEN })
    );
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    shared.set_watchers(fenced_on(), u32::MAX);
    let error = shared
        .lock()
        .prepare_frame(FIRST_TOKEN, frame(0, Some(1)))
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
        queue(&shared, FIRST_TOKEN, loose);
    }
    // Full before the watch: the counter's watch is not taken.
    let error = shared
        .lock()
        .prepare_frame(FIRST_TOKEN, frame(0, Some(1)))
        .unwrap_err();
    assert_eq!(error, QueueError::Full { token: FIRST_TOKEN });
    assert_eq!(shared.lock().watchers(fenced_on()), 0);
    assert_eq!(
        error.to_string(),
        format!("vi2 client {FIRST_TOKEN:#x} already has the most frames queued (15)")
    );
}

/// An abandoned frame gives back its watch; one reached at once has none to give.
#[test]
fn an_abandoned_frame_gives_back_its_watch() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    let mut state = shared.lock();
    let prepared = state.prepare_frame(FIRST_TOKEN, frame(0, Some(1))).unwrap();
    assert_eq!(state.watchers(fenced_on()), 1);
    state.abandon_frame(prepared);
    assert_eq!(state.watchers(fenced_on()), 0);
    assert!(!state.presents().has_work());
    // The next attempt succeeds.
    let prepared = state.prepare_frame(FIRST_TOKEN, frame(0, Some(1))).unwrap();
    state.commit_frame(prepared, Some(START), None);
    assert_eq!(state.watchers(fenced_on()), 1);
}

/// 202's removal drops a client's queued and shown frames and gives back each unreached
/// frame's watch; another client is untouched.
#[test]
fn unregistering_gives_back_the_watches() {
    let shared = fresh();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    register(&shared, SECOND_TOKEN, SECOND_BIT, None);
    queue(&shared, FIRST_TOKEN, frame(0, None));
    queue(&shared, SECOND_TOKEN, frame(0, Some(9)));
    turn(&shared, tick_at(1));
    queue(&shared, FIRST_TOKEN, frame(1, Some(9)));
    queue(&shared, FIRST_TOKEN, frame(2, Some(9)));
    assert_eq!(shared.lock().watchers(fenced_on()), 3);
    assert!(shared.lock().unregister_present(FIRST_TOKEN));
    assert!(!shared.lock().unregister_present(FIRST_TOKEN));
    let state = shared.lock();
    assert_eq!(state.watchers(fenced_on()), 1);
    assert!(state.presents().client(FIRST_TOKEN).is_none());
    assert_eq!(state.presents().client(SECOND_TOKEN).unwrap().queued().len(), 1);
}

/// 11's event reaches the client's entry; a token the table does not have changes nothing.
#[test]
fn the_notification_event_reaches_the_entry() {
    let mut table = Presents::new();
    table.register(FIRST_TOKEN, FIRST_BIT, None).unwrap();
    table.set_event(FIRST_TOKEN, event(4));
    table.set_event(SECOND_TOKEN, event(5));
    assert_eq!(table.client(FIRST_TOKEN).unwrap().event(), Some(event(4)));
    assert!(table.client(SECOND_TOKEN).is_none());
    assert_eq!(table.take_refresh(SECOND_TOKEN), Refresh::default());
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

fn saved_client(token: u64) -> SavedClient {
    SavedClient {
        token,
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
    // One client: token 0x21, bit 5, event (6, 7); one frame queued (slot 2, image 0x31, every
    // request field distinct, fenced on syncpoint 0x17 at 0x41, not reached, holds its slot);
    // the shown frame (slot 3, no fence, reached); released slot 1; raised; no held reply.
    let shared = shared_arc();
    register(&shared, 0x21, 5, Some(ObjectId::from_parts(6, 7)));
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
    queue(&shared, 0x21, NewFrame { slot: 1, ..shown });
    turn(&shared, tick_at(1));
    queue(&shared, 0x21, shown);
    turn(&shared, tick_at(2));
    queue(&shared, 0x21, behind);
    let mut expected = vec![b'S', b'Q', b'P', b'R', 0x01];
    expected.extend([0x01, 0x21, 0x05, 0x01, 0x06, 0x07]);
    // The queued frame: slot, image, address (0x200000 as a varint), width 102, height 202,
    // pitch 302, block height 7, format 1, shown 0, 0, 12, 22, layer 42, display 52, the fence
    // (some: 0x17, 0x41), not reached, holds.
    expected.extend([0x01, 0x02, 0x31, 0x80, 0x80, 0x80, 0x01]);
    expected.extend([0x66, 0xCA, 0x01, 0xAE, 0x02, 0x07, 0x01, 0x00, 0x00, 0x0C, 0x16]);
    expected.extend([0x2A, 0x34, 0x01, 0x17, 0x41, 0x00, 0x01]);
    // The shown frame: some; slot 3, image 0x32, address 0x100000, width 101, height 201,
    // pitch 301, 7, 1, shown 0, 0, 11, 21, layer 41, display 51, no fence, reached, holds.
    expected.extend([0x01, 0x03, 0x32, 0x80, 0x80, 0x40]);
    expected.extend([0x65, 0xC9, 0x01, 0xAD, 0x02, 0x07, 0x01, 0x00, 0x00, 0x0B, 0x15]);
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
    register(&source, FIRST_TOKEN, FIRST_BIT, Some(event(1)));
    queue(&source, FIRST_TOKEN, frame(0, None));
    turn(&source, tick_at(1));
    queue(&source, FIRST_TOKEN, frame(1, Some(4)));
    let bytes = source.save_presents(&no_server(&source)).unwrap();

    let target = shared_arc();
    target.lock().watch(fenced_on()).unwrap();
    target.restore_presents(&bytes, &no_server(&target)).unwrap();
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

/// A forged snapshot of one client, token 5, changed by `change`.
fn one_client(change: impl Fn(&mut SavedClient)) -> Vec<SavedClient> {
    let mut client = saved_client(5);
    change(&mut client);
    vec![client]
}

/// The client checks' faults, each with its error.
fn client_faults() -> Vec<(Vec<SavedClient>, PresentSnapshotError)> {
    vec![
        (
            (1..=u64::try_from(MAX_PRESENT_CLIENTS).unwrap() + 1)
                .map(saved_client)
                .collect(),
            PresentSnapshotError::TooManyClients {
                count: MAX_PRESENT_CLIENTS + 1,
            },
        ),
        (
            vec![saved_client(1), saved_client(0)],
            PresentSnapshotError::ZeroToken { index: 1 },
        ),
        (
            vec![saved_client(2), saved_client(2)],
            PresentSnapshotError::TokensNotAscending { index: 1 },
        ),
        (
            vec![saved_client(3), saved_client(2)],
            PresentSnapshotError::TokensNotAscending { index: 1 },
        ),
        (
            one_client(|c| c.available_bit = CLIENT_NOTIFICATIONS),
            PresentSnapshotError::NotificationBit { index: 0 },
        ),
        (
            one_client(|c| c.released = ALL_SLOTS + 1),
            PresentSnapshotError::Released { index: 0 },
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
    for (clients, expected) in client_faults().into_iter().chain(frame_faults()) {
        let shared = shared_arc();
        register(&shared, 0x99, 1, None);
        let before = shared.save_presents(&no_server(&shared)).unwrap();
        let error = shared
            .restore_presents(&snapshot(&clients), &no_server(&shared))
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(shared.save_presents(&no_server(&shared)).unwrap(), before);
    }
}

/// The edges of the restore checks, each accepted exactly at its limit: [`MAX_PRESENT_CLIENTS`]
/// clients, [`SLOTS`] frames, the last slot, the last syncpoint, the last notification bit,
/// every slot released, a reply behind one frame fewer than are queued, and an unreached frame
/// with exactly the waiters it needs.
#[test]
fn the_restore_checks_accept_their_edges() {
    let shared = shared_arc();
    let last_id = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    let last = SyncpointId::new(last_id).unwrap();
    shared.lock().watch(last).unwrap();
    let mut edge = saved_client(1);
    edge.available_bit = CLIENT_NOTIFICATIONS - 1;
    edge.queued = (0..SLOTS)
        .map(|slot| SavedFrame {
            holds_slot: false,
            ..saved_frame(slot, None, true)
        })
        .collect();
    edge.queued[0] = saved_frame(SLOTS - 1, Some((last_id, 3)), false);
    edge.released = ALL_SLOTS;
    edge.waiting = Some(SavedWaiting {
        token: None,
        ahead: u32::from(SLOTS) - 1,
    });
    let mut clients = vec![edge];
    clients.extend((2..=u64::try_from(MAX_PRESENT_CLIENTS).unwrap()).map(saved_client));
    shared
        .restore_presents(&snapshot(&clients), &no_server(&shared))
        .unwrap();
    assert_eq!(shared.lock().presents().clients().count(), MAX_PRESENT_CLIENTS);
    assert_eq!(shared.lock().presents().unreached_on(last), 1);
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
/// back is refused.
#[test]
fn saving_refuses_what_restoring_would() {
    let shared = shared_arc();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST_TOKEN, frame(0, Some(1)));
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
}

/// The vsync clock's snapshot keeps a deadline with no listener only while the table has work,
/// on save and on restore.
#[test]
fn the_clock_s_snapshot_takes_a_deadline_with_no_listener_only_while_a_present_waits() {
    let shared = shared_arc();
    register(&shared, FIRST_TOKEN, FIRST_BIT, None);
    queue(&shared, FIRST_TOKEN, frame(0, Some(1)));
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
