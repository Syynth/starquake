//! The `Gpu` through its threads, as gsv drives it.

use std::sync::mpsc;

use starquake_render::FenceValue;
use starquake_render::command::Command;

use super::*;
use crate::channel::ChannelError;
use crate::interface::CHANNEL_QUEUE_ENTRIES;
use crate::memory::GpuFault;

/// The homebrew-style stream of the end-to-end test: a release, two increments after it, a
/// present and a barrier, each waited for before the next call so the flight recorder's text is
/// in one order.
fn run_the_synthetic_stream(r: &Rig) -> ChannelId {
    r.syncpoints.watch(syncpoint());
    r.space();
    let channel = r.channel();
    let submission = Submission {
        entries: vec![r.entry(CMD, &releases(SEM, 0x77))],
        wait: None,
        increments: 2,
        wfi: false,
    };
    assert_eq!(r.gpu.submit(channel, submission), Submitted::Queued);
    r.wait_for(GpuCompletion::SyncpointChanged {
        id: syncpoint(),
        value: 2,
    });
    let frame: Vec<u32> = (0..64).map(|i| 0x0102_0304 * (i + 1)).collect();
    r.write(FRAME, &frame);
    r.gpu.present(present_request());
    wait_until("the present is recorded", || r.gpu.counters().presents == 1);
    r.gpu.barrier(Ticket(9));
    r.wait_for(GpuCompletion::GpuOrdered { ticket: Ticket(9) });
    channel
}

#[test]
fn a_synthetic_stream_produces_its_command_lists_and_completions_in_order() {
    let r = rig();
    run_the_synthetic_stream(&r);
    assert_eq!(r.word(SEM), 0x77);
    assert_eq!(
        r.pushed.all(),
        [
            GpuCompletion::SyncpointChanged {
                id: syncpoint(),
                value: 1
            },
            GpuCompletion::SyncpointChanged {
                id: syncpoint(),
                value: 2
            },
            GpuCompletion::GpuOrdered { ticket: Ticket(9) },
        ]
    );
    // One list: the frame's upload and the present that closes it.
    let lists = r.recording().lists();
    assert_eq!(lists.len(), 1);
    assert!(matches!(
        lists[0].1.commands.last(),
        Some(Command::Present(desc)) if desc.frame == 0 && desc.src_rect.width == FRAME_SIZE
    ));
    insta::assert_snapshot!(r.recording().snapshot());
    r.gpu.stop().unwrap();
}

#[test]
fn gpu_txt_holds_the_runs_events_and_the_counters_count_it() {
    let r = rig();
    let channel = run_the_synthetic_stream(&r);
    r.gpu.stop().unwrap();
    let counters = r.gpu.counters();
    assert_eq!(counters.submissions, 1);
    assert_eq!(counters.increments, 2);
    assert_eq!(counters.presents_pushed, 1);
    assert_eq!(counters.presents, 1);
    assert_eq!(counters.ordered, 1);
    assert_eq!(counters.entries_refused, 0);
    // The release and both increments, and the barrier's marker.
    assert_eq!(counters.fenced_actions, 4);
    assert_eq!(counters.lists_submitted, 1);
    let expected = format!(
        "0 address space {space} made\n\
         1 channel {channel} made in address space {space}, incrementing syncpoint {SYNCPOINT}\n\
         2 submission on channel {channel}: 1 entries, 2 increments: queued\n\
         3 performed 1 releases, 0 increments, 0 markers, no waiter latched\n\
         4 performed 0 releases, 1 increments, 0 markers, no waiter latched\n\
         5 performed 0 releases, 1 increments, 0 markers, no waiter latched\n\
         6 {present}\n\
         7 barrier, ticket 9\n\
         8 pushed: ticket 9 done\n\
         9 performed 0 releases, 0 increments, 1 markers, no waiter latched\n",
        space = SPACE.0,
        channel = channel.0,
        present = RingEntry::Present(present_request()),
    );
    assert_eq!(r.gpu.flight_recorder(), expected);
}

#[test]
fn full_and_queue_room_end_to_end() {
    let r = rig();
    r.space();
    let channel = r.channel();
    let mut parked = PushBuffer::new();
    acquire(&mut parked, SEM, 1);
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &parked.into_words())]));
    wait_until("the channel parks", || {
        r.gpu.counters().acquires_parked == 1
    });
    // The parked submission's one entry has started, so the queue is empty: a submission of
    // exactly the bound fits, and one entry more does not.
    let mut bind = PushBuffer::new();
    bind_3d(&mut bind);
    let one = r.entry(CMD2, &bind.into_words());
    let filling = plain(vec![one; CHANNEL_QUEUE_ENTRIES]);
    assert_eq!(r.gpu.submit(channel, filling), Submitted::Queued);
    assert_eq!(r.gpu.submit(channel, plain(vec![one])), Submitted::Full);
    assert_eq!(r.gpu.counters().submissions_full, 1);
    assert_eq!(r.pushed.all(), []);
    // The CPU releases the acquire; the front's poll sees it, the channel runs, and the room
    // the refused submission waits for is reported.
    r.write(SEM, &[1]);
    r.wait_for(GpuCompletion::GpuQueueRoom { channel });
    assert_eq!(r.gpu.submit(channel, plain(vec![one])), Submitted::Queued);
    r.gpu.stop().unwrap();
}

#[test]
fn an_ordered_unmap_is_reported_after_the_work_queued_before_it() {
    let r = rig();
    r.space();
    let channel = r.channel();
    let mut stream = PushBuffer::new();
    acquire(&mut stream, SEM, 1);
    release(&mut stream, SEM2, 5);
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &stream.into_words())]));
    wait_until("the channel parks", || {
        r.gpu.counters().acquires_parked == 1
    });
    let all = PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap();
    r.gpu
        .update_vm(SPACE, VmUpdate::Unmap { range: all }, Some(Ticket(0x31)));
    // A parked channel leaves the GPU idle: only the guest can release it. The update waits.
    wait_until("the GPU is idle", || r.gpu.is_idle());
    assert_eq!(r.pushed.all(), []);
    assert_eq!(r.vm.mappings().len(), 1);
    r.write(SEM, &[1]);
    let done = GpuCompletion::GpuOrdered {
        ticket: Ticket(0x31),
    };
    r.wait_for(done);
    // The release after the acquire was written before the update was reported, and the table
    // was changed by then.
    assert_eq!(r.pushed.probe_at(done), Some(5));
    assert_eq!(r.vm.mappings(), []);
    r.gpu.stop().unwrap();
}

#[test]
fn a_channel_made_and_used_at_once_is_known_to_the_back() {
    let r = rig();
    r.space();
    let channel = r.channel();
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(SEM, 0x5a))]));
    wait_until("the release is written", || r.word(SEM) == 0x5a);
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.counters().entries_refused, 0);
}

#[test]
fn a_release_is_written_through_the_back_threads_memory_of_its_channels_space() {
    // Two spaces map the same GPU pages to different CPU pages: the channel's release lands in
    // its own space's.
    let r = rig();
    let other_cpu = CPU_BASE + 16 * PAGE_SIZE;
    let other = Arc::new(GpuVm::new());
    other
        .map(
            PageRange::new(GPU_BASE, 16 * PAGE_SIZE).unwrap(),
            Target::Guest { cpu_va: CPU_BASE },
            Kind(0),
        )
        .unwrap();
    other
        .map(
            PageRange::new(SEM & !(PAGE_SIZE - 1), PAGE_SIZE).unwrap(),
            Target::Guest { cpu_va: other_cpu },
            Kind(0),
        )
        .unwrap();
    r.space();
    r.gpu.create_address_space(OTHER_SPACE, other);
    let channel = r
        .gpu
        .create_channel(ChannelConfig {
            space: OTHER_SPACE,
            syncpoint: syncpoint(),
        })
        .unwrap();
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(SEM, 0x6b))]));
    let landed = other_cpu + (SEM & (PAGE_SIZE - 1));
    let read = |va: u64| {
        let mut bytes = [0; 4];
        r.guest.read(va, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    };
    wait_until("the release is written", || read(landed) == 0x6b);
    assert_eq!(r.word(SEM), 0, "the first space's page is untouched");
    r.gpu.stop().unwrap();
}

#[test]
fn presents_read_through_the_first_address_space_made() {
    let r = rig();
    r.space();
    // A second space that maps nothing: a present read through it would be refused.
    r.gpu
        .create_address_space(OTHER_SPACE, Arc::new(GpuVm::new()));
    r.gpu.present(present_request());
    wait_until("the present is recorded", || r.gpu.counters().presents == 1);
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.counters().entries_refused, 0);
}

#[test]
fn a_present_with_no_address_space_is_refused_and_counted() {
    let r = rig();
    r.gpu.present(present_request());
    wait_until("the present is refused", || {
        r.gpu.counters().entries_refused == 1
    });
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.counters().presents, 0);
}

#[test]
fn the_gpu_is_busy_while_a_list_is_in_flight() {
    let r = rig_on(true);
    r.space();
    r.gpu.present(present_request());
    wait_until("the present's list is submitted", || {
        r.late().submitted().len() == 1
    });
    r.gpu.barrier(Ticket(4));
    wait_until("the barrier's marker is taken", || {
        r.gpu.back_status().taken == 2
    });
    assert!(!r.gpu.is_idle());
    assert_eq!(r.pushed.all(), []);
    r.late().finish(FenceValue::new(1));
    r.wait_for(GpuCompletion::GpuOrdered { ticket: Ticket(4) });
    wait_until("the GPU is idle", || r.gpu.is_idle());
    r.gpu.stop().unwrap();
}

#[test]
fn the_gpu_is_idle_once_its_work_is_done_and_busy_before() {
    let r = rig();
    wait_until("a new GPU is idle", || r.gpu.is_idle());
    r.space();
    let channel = r.channel();
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(SEM, 1))]));
    wait_until("the release is written", || r.word(SEM) == 1);
    wait_until("the GPU is idle again", || r.gpu.is_idle());
    r.gpu.stop().unwrap();
}

#[test]
fn a_destroyed_channel_is_forgotten_by_the_back() {
    let r = rig();
    r.space();
    let channel = r.channel();
    wait_until("the back knows the channel", || {
        r.gpu.back_status().channels == 1
    });
    r.gpu.destroy_channel(channel);
    wait_until("the back forgets the channel", || {
        r.gpu.back_status().channels == 0
    });
    assert_eq!(r.gpu.bind_class(channel, clc797::CLASS), Err(GpuError));
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.back_status().refused_controls, 0);
}

#[test]
fn bind_class_takes_a_channel_made_and_not_destroyed() {
    let r = rig();
    r.space();
    let channel = r.channel();
    assert_eq!(r.gpu.bind_class(channel, clc797::CLASS), Ok(()));
    assert_eq!(
        r.gpu.bind_class(ChannelId(channel.0 + 1), clc797::CLASS),
        Err(GpuError)
    );
    r.gpu.stop().unwrap();
}

#[test]
fn a_channel_in_a_space_never_made_is_refused_and_recorded() {
    let r = rig();
    let refused = r.gpu.create_channel(ChannelConfig {
        space: SPACE,
        syncpoint: syncpoint(),
    });
    assert_eq!(refused, Err(GpuError));
    assert_eq!(
        r.gpu.recorder().events(),
        [GpuEvent::ChannelRefused { space: SPACE }]
    );
    r.gpu.stop().unwrap();
}

#[test]
fn an_address_space_made_twice_keeps_the_first() {
    let r = rig();
    r.space();
    r.gpu.create_address_space(SPACE, Arc::new(GpuVm::new()));
    let channel = r.channel();
    // The first table maps `SEM`; the second maps nothing.
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(SEM, 0x2c))]));
    wait_until("the release is written", || r.word(SEM) == 0x2c);
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.back_status().spaces, 1);
    assert_eq!(r.gpu.back_status().refused_controls, 0);
    assert_eq!(
        r.gpu.recorder().events()[..1],
        [GpuEvent::SpaceMade { space: SPACE }]
    );
}

#[test]
fn a_stopped_channels_error_reaches_gsv_and_gpu_txt() {
    let r = rig();
    r.space();
    let channel = r.channel();
    // An entry past the mapped pages: its first word cannot be read.
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    r.gpu.submit(channel, plain(vec![raw_entry(unmapped, 4)]));
    wait_until("the channel stops", || {
        r.gpu.channel_error(channel).is_some()
    });
    let error = ChannelError::ReadFault {
        va: unmapped,
        fault: GpuFault::Unmapped { va: unmapped },
    };
    assert_eq!(r.gpu.channel_error(channel), Some(error));
    r.gpu.stop().unwrap();
    assert_eq!(r.gpu.counters().channels_stopped, 1);
    let last = format!("channel {} stopped: {error}\n", channel.0);
    assert!(
        r.gpu.flight_recorder().ends_with(&last),
        "{}",
        r.gpu.flight_recorder()
    );
}

#[test]
fn both_threads_end_when_the_gpu_stops() {
    let r = rig();
    r.space();
    let channel = r.channel();
    r.gpu
        .submit(channel, plain(vec![r.entry(CMD, &releases(SEM, 3))]));
    assert!(r.gpu.is_running());
    let gpu = Arc::new(r.gpu);
    let stopping = Arc::clone(&gpu);
    let (sent, stopped) = mpsc::channel();
    std::thread::spawn(move || {
        sent.send(stopping.stop().map_err(|e| e.to_string()))
            .unwrap();
    });
    let result = stopped
        .recv_timeout(PATIENCE)
        .expect("the GPU's threads did not end in time");
    assert_eq!(result, Ok(()));
    assert!(!gpu.is_running());
    // A second stop does nothing.
    assert!(gpu.stop().is_ok());
}

#[test]
fn dropping_the_gpu_ends_its_threads() {
    let r = rig();
    r.space();
    let (sent, dropped) = mpsc::channel();
    std::thread::spawn(move || {
        drop(r);
        sent.send(()).unwrap();
    });
    dropped
        .recv_timeout(PATIENCE)
        .expect("dropping the GPU did not end its threads in time");
}
