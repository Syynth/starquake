//! The null GPU front in both modes, and the replay of recorded notes, over a [`Graphics`] of
//! their own (no kernel). The kernel host and the wiring through a real dispatcher are
//! `hle/tests/graphics.rs`'s.

use starquake_gpu::interface::{Fence, RegionId};
use starquake_gsv::note::{SYNCPT_PAGE_ACKNOWLEDGED, SYNCPT_PAGE_ANSWERED};
use starquake_memory::PageRange;

use super::*;

mod vsync;

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

/// A syncpoint nothing else in these tests touches, and its neighbour.
const SP: u8 = 40;
const NEIGHBOUR: u8 = 41;

/// A graphics server in `mode`, and its queue emptied by the caller as it goes.
fn graphics(mode: FrontMode) -> Graphics {
    Graphics::new(mode, None)
}

/// What the queue holds now, taken.
fn queued(graphics: &Graphics) -> Vec<Completion> {
    graphics.shared().queue().take_all()
}

fn gpu(completion: GpuCompletion) -> Completion {
    Completion::Gpu(completion)
}

fn config(sp: u8) -> ChannelConfig {
    ChannelConfig {
        space: AsId(1),
        syncpoint: id(sp),
    }
}

fn submission(increments: u32, wait: Option<Fence>) -> Submission {
    Submission {
        entries: Vec::new(),
        wait,
        increments,
        wfi: true,
    }
}

fn unmap() -> VmUpdate {
    VmUpdate::Unmap {
        range: PageRange::new(0x8000_0000, 0x1000).unwrap(),
    }
}

/// In a run an ordered update with a ticket and a barrier are done at once, each pushing its
/// ticket in the order given; an update with no ticket pushes nothing. In a replay nothing is
/// pushed.
#[test]
fn ordered_updates_and_barriers_are_done_at_once_in_a_run() {
    let live = graphics(FrontMode::Live);
    let front = live.front();
    front.update_vm(AsId(1), unmap(), Some(Ticket(0x31)));
    front.update_vm(AsId(1), unmap(), None);
    front.barrier(Ticket(0x32));
    let destroy = VmUpdate::DestroySparseRegion {
        range: PageRange::new(0x8000_0000, 0x1000).unwrap(),
        region: RegionId(3),
    };
    front.update_vm(AsId(2), destroy, Some(Ticket(0x33)));
    assert_eq!(
        queued(&live),
        [
            gpu(GpuCompletion::GpuOrdered {
                ticket: Ticket(0x31)
            }),
            gpu(GpuCompletion::GpuOrdered {
                ticket: Ticket(0x32)
            }),
            gpu(GpuCompletion::GpuOrdered {
                ticket: Ticket(0x33)
            }),
        ]
    );

    let replay = graphics(FrontMode::Replay);
    replay
        .front()
        .update_vm(AsId(1), unmap(), Some(Ticket(0x31)));
    replay.front().barrier(Ticket(0x32));
    assert_eq!(queued(&replay), []);
}

/// In a run a submission's increments are performed at once on the channel's syncpoint, with one
/// push for a watched syncpoint however many they are, and the answer is `Queued`, never `Full`,
/// however many entries; in a replay nothing is performed. The neighbour is untouched.
#[test]
fn a_submission_s_increments_are_performed_at_once() {
    for mode in [FrontMode::Live, FrontMode::Replay] {
        let graphics = graphics(mode);
        let front = graphics.front();
        let syncpoints = graphics.shared().syncpoints();
        let channel = front.create_channel(config(SP)).unwrap();
        syncpoints.watch(id(SP));
        let mut big = submission(0x10, None);
        big.entries = vec![starquake_gsv::nvsrv::gpu::GpfifoEntry(7); 0x1_0000];
        assert_eq!(front.submit(channel, big), Submitted::Queued);
        let (value, pushed) = match mode {
            FrontMode::Live => (
                0x10,
                vec![gpu(GpuCompletion::SyncpointChanged {
                    id: id(SP),
                    value: 0x10,
                })],
            ),
            FrontMode::Replay => (0, Vec::new()),
        };
        assert_eq!(syncpoints.read(id(SP)), value, "{mode:?}");
        assert_eq!(queued(&graphics), pushed, "{mode:?}");
        assert_eq!(syncpoints.read(id(NEIGHBOUR)), 0);
        // No increments: nothing changes, nothing is pushed.
        assert_eq!(
            front.submit(channel, submission(0, None)),
            Submitted::Queued
        );
        assert_eq!(syncpoints.read(id(SP)), value);
        assert_eq!(queued(&graphics), []);
    }
}

/// A run's front does not wait for a submission's fence: the increments are performed at once
/// though the fence is not reached (a listed divergence from a GPU).
#[test]
fn a_submission_s_wait_fence_is_not_waited_for() {
    let graphics = graphics(FrontMode::Live);
    let front = graphics.front();
    let syncpoints = graphics.shared().syncpoints();
    let channel = front.create_channel(config(SP)).unwrap();
    let fence = Fence {
        id: id(NEIGHBOUR),
        value: 5,
    };
    assert!(!syncpoints.reached(fence));
    assert_eq!(
        front.submit(channel, submission(2, Some(fence))),
        Submitted::Queued
    );
    assert_eq!(syncpoints.read(id(SP)), 2);
}

/// A submission on a channel the front did not make, or no longer holds, performs nothing and
/// answers `Queued`.
#[test]
fn a_submission_on_an_unknown_channel_does_nothing() {
    let graphics = graphics(FrontMode::Live);
    let front = graphics.front();
    let syncpoints = graphics.shared().syncpoints();
    let made = front.create_channel(config(SP)).unwrap();
    let never = ChannelId(made.0 + 1);
    assert_eq!(front.submit(never, submission(2, None)), Submitted::Queued);
    front.destroy_channel(made);
    assert_eq!(front.submit(made, submission(2, None)), Submitted::Queued);
    assert_eq!(syncpoints.read(id(SP)), 0);
    assert_eq!(queued(&graphics), []);
}

/// Channels are numbered from [`FIRST_CHANNEL`] up, the same in both modes; a class binds on a
/// channel the front holds and not on another; a destroyed channel goes; and once the numbers
/// are used up making one fails and changes nothing.
#[test]
fn channels_count_up_from_the_first_and_go_when_destroyed() {
    for mode in [FrontMode::Live, FrontMode::Replay] {
        let graphics = graphics(mode);
        let front = graphics.front();
        let first = front.create_channel(config(SP)).unwrap();
        let second = front.create_channel(config(NEIGHBOUR)).unwrap();
        assert_eq!(
            (first, second),
            (ChannelId(FIRST_CHANNEL), ChannelId(FIRST_CHANNEL + 1))
        );
        assert_eq!(front.channels(), [(first, id(SP)), (second, id(NEIGHBOUR))]);
        assert_eq!(front.bind_class(first, 0xC797), Ok(()));
        assert_eq!(
            front.bind_class(ChannelId(FIRST_CHANNEL + 2), 0xC797),
            Err(GpuError)
        );
        front.destroy_channel(first);
        assert_eq!(front.channels(), [(second, id(NEIGHBOUR))]);
        assert_eq!(front.bind_class(first, 0xC797), Err(GpuError));
        // A destroyed number is not handed out again.
        assert_eq!(
            front.create_channel(config(SP)),
            Ok(ChannelId(FIRST_CHANNEL + 2))
        );
    }
    let graphics = graphics(FrontMode::Live);
    let front = graphics.front();
    front.channels.lock().unwrap().next = u32::MAX - 1;
    assert_eq!(
        front.create_channel(config(SP)),
        Ok(ChannelId(u32::MAX - 1))
    );
    assert_eq!(front.create_channel(config(NEIGHBOUR)), Err(GpuError));
    assert_eq!(front.channels(), [(ChannelId(u32::MAX - 1), id(SP))]);
}

/// `GpuOrderedDone` and `QueueRoom` are fed back as the completions they record.
#[test]
fn ordered_and_room_notes_are_fed_back_as_their_completions() {
    let graphics = graphics(FrontMode::Replay);
    graphics
        .replay(Note::GpuOrderedDone {
            ticket: Ticket(0x61),
        })
        .unwrap();
    graphics
        .replay(Note::QueueRoom {
            channel: ChannelId(0x62),
        })
        .unwrap();
    assert_eq!(
        queued(&graphics),
        [
            gpu(GpuCompletion::GpuOrdered {
                ticket: Ticket(0x61)
            }),
            gpu(GpuCompletion::GpuQueueRoom {
                channel: ChannelId(0x62)
            }),
        ]
    );
}

/// A `SyncpointObserved` note brings the counter to its value when that is ahead by the 32-bit
/// wrapping comparison (up to 2^31 - 1 ahead), and is pushed as it is: one entry, watched or
/// not (the counter's own push of a watched id folds into it); a value not ahead (equal, behind,
/// or exactly 2^31 away) leaves the counter and is pushed as it is.
#[test]
fn a_syncpoint_note_brings_the_counter_to_its_value_when_ahead() {
    const START: u32 = 0x10;
    let half = 1_u32 << 31;
    let cases = [
        // (value, watched, the counter after)
        (START + 5, true, START + 5),
        (START + 5, false, START + 5),
        (
            START.wrapping_add(half - 1),
            true,
            START.wrapping_add(half - 1),
        ),
        (START.wrapping_add(half), true, START),
        (START, true, START),
        (START - 1, true, START),
        (START - 1, false, START),
    ];
    for (value, watched, after) in cases {
        let graphics = graphics(FrontMode::Replay);
        let syncpoints = graphics.shared().syncpoints();
        syncpoints.increment_by(id(SP), START);
        if watched {
            syncpoints.watch(id(SP));
        }
        graphics
            .replay(Note::SyncpointObserved { id: id(SP), value })
            .unwrap();
        assert_eq!(syncpoints.read(id(SP)), after, "{value:#x} {watched}");
        assert_eq!(
            queued(&graphics),
            [gpu(GpuCompletion::SyncpointChanged { id: id(SP), value })],
            "{value:#x} {watched}"
        );
        assert_eq!(syncpoints.read(id(NEIGHBOUR)), 0);
    }
}

/// The shared page's notes are what a page turn writes: each is refused with its code, in plain
/// words, and nothing is pushed.
#[test]
fn the_page_s_notes_are_not_fed_back() {
    let graphics = graphics(FrontMode::Replay);
    let answered = Note::SyncptPageAnswered {
        id: id(SP),
        value: 3,
    };
    let acknowledged = Note::SyncptPageAcknowledged { groups: 1 };
    assert_eq!(
        graphics.replay(answered),
        Err(ReplayError::NotAnInput {
            code: SYNCPT_PAGE_ANSWERED
        })
    );
    assert_eq!(
        graphics.replay(acknowledged),
        Err(ReplayError::NotAnInput {
            code: SYNCPT_PAGE_ACKNOWLEDGED
        })
    );
    assert_eq!(queued(&graphics), []);
    assert_eq!(graphics.shared().syncpoints().read(id(SP)), 0);
    assert_eq!(
        ReplayError::NotAnInput { code: 2 }.to_string(),
        "a note with code 2 is what the graphics server wrote, not something it was told, so it is \
         not fed back"
    );
}

/// The notes record for the root's number, which the wiring gives the shared state.
#[test]
fn the_notes_are_recorded_for_the_root() {
    struct Notes(Vec<EventKind>);
    impl TimerHost for Notes {
        fn record(&mut self, kind: EventKind) {
            self.0.push(kind);
        }
        fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
            Ok(())
        }
        fn signal_event(&mut self, _: ObjectId) -> SvcResult<()> {
            Ok(())
        }
    }
    let graphics = graphics(FrontMode::Live);
    graphics.front().barrier(Ticket(0x71));
    let mut host = Notes(Vec::new());
    let taken = graphics.turn(&mut host, HostNanos(0));
    assert_eq!(
        taken,
        [gpu(GpuCompletion::GpuOrdered {
            ticket: Ticket(0x71)
        })]
    );
    assert_eq!(
        host.0,
        [Note::GpuOrderedDone {
            ticket: Ticket(0x71)
        }
        .event(GSV_ROOT)]
    );
    assert_eq!(graphics.deadline(), None);
}

/// A fresh graphics server is idle; something queued makes it not idle, and the turn that drains
/// it makes it idle again.
#[test]
fn the_graphics_server_is_idle_only_with_nothing_queued() {
    struct Quiet;
    impl TimerHost for Quiet {
        fn record(&mut self, _: EventKind) {}
        fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
            Ok(())
        }
        fn signal_event(&mut self, _: ObjectId) -> SvcResult<()> {
            Ok(())
        }
    }
    let graphics = graphics(FrontMode::Live);
    assert!(graphics.is_idle());
    graphics.front().barrier(Ticket(0x81));
    assert!(!graphics.is_idle());
    graphics.turn(&mut Quiet, HostNanos(0));
    assert!(graphics.is_idle());
}

/// `is_idle` waits for gsv's shared lock, which a drain holds while it signals: asked while
/// another thread holds the lock, it answers only once the lock is let go.
#[test]
fn is_idle_waits_for_a_drain_in_progress() {
    use std::sync::mpsc;
    use std::time::Duration;

    let graphics = Arc::new(graphics(FrontMode::Live));
    let held = graphics.shared().lock();
    let (answered, answer) = mpsc::channel();
    let asking = Arc::clone(&graphics);
    let thread = std::thread::spawn(move || {
        answered.send(asking.is_idle()).unwrap();
    });
    assert_eq!(
        answer.recv_timeout(Duration::from_millis(200)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    drop(held);
    assert_eq!(answer.recv_timeout(Duration::from_secs(10)), Ok(true));
    thread.join().unwrap();
}
