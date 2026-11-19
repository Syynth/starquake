//! The back thread's loop, one pass at a time and with no thread: where a control is applied
//! against the ring entries around it (`back_thread`'s module docs).

use crate::back::fence::{FrontWake, Report, WaiterTable};
use crate::ring::{PopError, Producer, Release, ReleaseOp};

use super::*;

/// The channel the controls name: not 0, so a hard-coded id fails.
const CHANNEL: ChannelId = ChannelId(0x41);

/// A front that ignores the back's reports.
struct NoFront;

impl FrontWake for NoFront {
    fn wake(&self, _report: &Report) {}
}

struct Loop {
    back: Back,
    producer: Producer,
    guest: Arc<FlatMemory>,
}

/// A back over the recording backend and the rig's guest memory, with an empty inbox.
fn back_loop() -> Loop {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let guest = Arc::new(FlatMemory::new(CPU_BASE, size));
    let pushed = Arc::new(Pushed::default());
    let syncpoints = Arc::new(Syncpoints::new(Arc::<Pushed>::clone(&pushed)));
    let GpuRing { producer, consumer } = GpuRing::new(1 << 12).unwrap();
    let backend: Arc<dyn Backend> = Arc::new(RecordingBackend::new(caps(), 1 << 30));
    let env = FenceEnv {
        syncpoints,
        waiters: Arc::new(WaiterTable::new()),
        completions: pushed,
        clock: Arc::new(Clock::default()),
        front: Arc::new(NoFront),
    };
    let fences = Fences::new(Arc::clone(&backend), env, Mode::Fenced);
    let exec = Executor::new(
        backend,
        fences,
        Box::new(NoShaders),
        Arc::new(OrderedTable::new()),
        ExecConfig::default(),
    );
    let back = Back {
        exec,
        inbox: Arc::new(BackInbox::new(consumer.waker())),
        ring: consumer,
        shared: Arc::new(BackShared::default()),
        channels: BTreeSet::new(),
        spaces: BTreeSet::new(),
        taken: 0,
        wait_failures: 0,
        refused_controls: 0,
    };
    Loop {
        back,
        producer,
        guest,
    }
}

impl Loop {
    /// The controls that make `SPACE` (presents read through it) and [`CHANNEL`] in it.
    fn space_and_channel(&self) -> [Control; 2] {
        let memory: Arc<dyn GpuMemory> = Arc::new(VmMemory::new(
            table(),
            Arc::<FlatMemory>::clone(&self.guest),
            Arc::new(Syncpoints::new(Arc::new(Pushed::default()))),
        ));
        [
            Control::AddSpace {
                space: SPACE,
                memory,
                watch: Arc::new(WriteWatch::new()),
                present: true,
            },
            Control::AddChannel {
                channel: CHANNEL,
                space: SPACE,
            },
        ]
    }

    /// Pushes and publishes a release of `payload` at `SEM` issued by [`CHANNEL`].
    fn push_release(&mut self, payload: u32) {
        let entry = RingEntry::Release {
            channel: CHANNEL,
            release: Release {
                address: SEM,
                payload: u64::from(payload),
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false,
                },
                op: ReleaseOp::Write,
            },
        };
        self.producer.push(&entry).unwrap();
        self.producer.publish();
    }

    fn step(&mut self) {
        assert!(self.back.step(PATIENCE).unwrap());
    }

    fn word(&self) -> u32 {
        cpu_word(&self.guest, SEM)
    }
}

#[test]
fn a_forget_waits_for_the_entries_pushed_before_it() {
    // The front dropped the channel after pushing one entry: entry 0 is applied with the
    // channel known, entry 1 (pushed after the drop) finds it forgotten.
    let mut l = back_loop();
    for control in l.space_and_channel() {
        l.back.inbox.post(control);
    }
    l.push_release(0x55);
    l.back.inbox.post(Control::Forget {
        channel: CHANNEL,
        after: 1,
    });
    l.push_release(0x66);
    l.step();
    assert_eq!(l.word(), 0x55);
    assert_eq!(l.back.channels.len(), 1, "not forgotten before entry 1");
    l.step();
    assert_eq!(l.word(), 0x55, "entry 1 was refused");
    let counters = l.back.exec.counters();
    assert_eq!((counters.applied, counters.failed), (1, 1));
    assert!(l.back.channels.is_empty());
    assert_eq!(l.back.shared.status().channels, 0);
}

#[test]
fn a_forget_due_at_once_is_applied_before_the_next_entry() {
    // Dropped before it pushed anything: its first entry already finds it forgotten.
    let mut l = back_loop();
    for control in l.space_and_channel() {
        l.back.inbox.post(control);
    }
    l.back.inbox.post(Control::Forget {
        channel: CHANNEL,
        after: 0,
    });
    l.push_release(0x55);
    l.step();
    assert_eq!(l.word(), 0);
    assert_eq!(l.back.exec.counters().failed, 1);
}

#[test]
fn controls_posted_after_a_forget_that_is_not_due_wait_behind_it() {
    let mut l = back_loop();
    let [space, channel] = l.space_and_channel();
    l.back.inbox.post(space);
    l.back.inbox.post(Control::Forget {
        channel: ChannelId(0x99),
        after: 1,
    });
    l.back.inbox.post(channel);
    l.back.apply_controls();
    assert_eq!(l.back.spaces.len(), 1);
    assert!(l.back.channels.is_empty());
    l.push_release(0x10);
    // The forget is due before entry 1, not entry 0: the channel is added after entry 0 is
    // applied, so entry 0 is refused.
    l.step();
    assert_eq!(l.back.exec.counters().failed, 1);
    l.back.apply_controls();
    assert_eq!(l.back.channels.len(), 1);
}

#[test]
fn a_channel_posted_after_its_entry_was_pushed_is_known_before_the_entry_is_applied() {
    // The race of `a_channel_made_and_used_at_once_is_known_to_the_back`, made exact: the back
    // has looked at its inbox and found nothing, and is about to sleep on the ring, when the
    // channel is posted and its entry pushed. It must look at its inbox again after the pop.
    let l = back_loop();
    let controls = Mutex::new(Some(l.space_and_channel()));
    let Loop {
        mut back,
        producer,
        guest,
    } = l;
    let inbox = Arc::clone(&back.inbox);
    // The producer, pushed from inside the hook and kept alive in the slot past the pop.
    let slot = Arc::new(Mutex::new(producer));
    let pushing = Arc::clone(&slot);
    back.ring.set_after_last_look(Box::new(move || {
        let Some(posted) = controls.lock().unwrap().take() else {
            return;
        };
        for control in posted {
            inbox.post(control);
        }
        let entry = RingEntry::Release {
            channel: CHANNEL,
            release: Release {
                address: SEM,
                payload: 0x77,
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false,
                },
                op: ReleaseOp::Write,
            },
        };
        let mut producer = pushing.lock().unwrap();
        producer.push(&entry).unwrap();
        producer.publish();
    }));
    assert!(back.step(PATIENCE).unwrap());
    assert_eq!(back.exec.counters().failed, 0);
    assert_eq!(cpu_word(&guest, SEM), 0x77);
    drop(slot);
}

#[test]
fn a_space_posted_with_present_set_is_the_one_presents_read() {
    let mut l = back_loop();
    let [space, _] = l.space_and_channel();
    l.back.inbox.post(space);
    l.producer
        .push(&RingEntry::Present(present_request()))
        .unwrap();
    l.producer.publish();
    l.step();
    assert_eq!(l.back.exec.counters().presents, 1);
}

#[test]
fn a_space_posted_without_present_set_is_not_read_by_presents() {
    let mut l = back_loop();
    let [space, _] = l.space_and_channel();
    let Control::AddSpace {
        space: id,
        memory,
        watch,
        ..
    } = space
    else {
        unreachable!("the first control adds the space");
    };
    l.back.inbox.post(Control::AddSpace {
        space: id,
        memory,
        watch,
        present: false,
    });
    l.producer
        .push(&RingEntry::Present(present_request()))
        .unwrap();
    l.producer.publish();
    l.step();
    assert_eq!(l.back.exec.counters().presents, 0);
    assert_eq!(l.back.exec.counters().failed, 1);
}

#[test]
fn a_refused_control_is_counted_and_the_loop_goes_on() {
    let mut l = back_loop();
    l.back.inbox.post(Control::AddChannel {
        channel: CHANNEL,
        space: SPACE,
    });
    let [space, channel] = l.space_and_channel();
    l.back.inbox.post(space);
    l.back.inbox.post(channel);
    l.back.apply_controls();
    // The first add named a space not yet added; the second is the one that took.
    assert_eq!(l.back.refused_controls, 1);
    assert_eq!(l.back.channels.len(), 1);
}

#[test]
fn the_loop_ends_when_the_producer_is_gone_after_the_last_entry() {
    let mut l = back_loop();
    for control in l.space_and_channel() {
        l.back.inbox.post(control);
    }
    l.push_release(0x31);
    let Loop {
        mut back,
        producer,
        guest,
    } = l;
    drop(producer);
    assert!(back.step(PATIENCE).unwrap(), "the entry first");
    assert_eq!(cpu_word(&guest, SEM), 0x31);
    assert!(!back.step(PATIENCE).unwrap());
    assert_eq!(back.shared.status().taken, 1);
}

#[test]
fn a_stop_ends_the_loop_at_its_next_pass() {
    let mut l = back_loop();
    l.back.inbox.stop();
    assert!(!l.back.step(PATIENCE).unwrap());
}

#[test]
fn a_corrupt_ring_ends_the_loop_with_its_error() {
    let mut l = back_loop();
    l.producer.push_raw(&[u64::MAX]);
    l.producer.publish();
    let error = l.back.step(PATIENCE).unwrap_err();
    assert!(matches!(error, BackError::Ring(PopError::Corrupt(_))), "{error}");
}
