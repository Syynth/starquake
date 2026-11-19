//! Scheduling: parked channels skipped, round robin with a quantum, no starvation; routing of
//! what a channel executes; channels that stop; destroyed channels.

use super::*;
use crate::macros::Engine;

/// A stream that binds 3D and then writes `values` to scratch `index` one immediate each: one
/// word per write.
fn immediates(index: u32, values: impl IntoIterator<Item = u16>) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    for value in values {
        pb.immediate(0, scratch(index), value).unwrap();
    }
    pb.into_words()
}

/// The channels of the `Write` entries in `texts`, collapsed into runs: `(channel, count)`.
fn runs_of_writes(texts: &[String]) -> Vec<(u32, usize)> {
    let mut runs: Vec<(u32, usize)> = Vec::new();
    for text in texts.iter().filter(|t| t.starts_with("write ")) {
        let channel: u32 = text
            .split_whitespace()
            .find_map(|field| field.strip_prefix("channel="))
            .unwrap()
            .parse()
            .unwrap();
        match runs.last_mut() {
            Some((last, count)) if *last == channel => *count += 1,
            _ => runs.push((channel, 1)),
        }
    }
    runs
}

#[test]
fn a_parked_channel_does_not_block_another() {
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    acquire(&mut pb, SEM, AcquireOp::Equal, 1);
    pb.methods(MethodMode::Incrementing, 0, scratch(0), &[0xa1])
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.settle();
    let texts = r.take();
    assert!(has(&texts, &scratch_text(b, 1, 0xb1)));
    assert!(!has(&texts, &scratch_text(a, 0, 0xa1)));
    assert_eq!(r.front.counters().parks, 1);
    // B goes on submitting and running while A waits.
    r.submit(b, vec![r.entry(CMD_C, &write_scratch(2, 0xb2))]);
    r.settle();
    let texts = r.take();
    assert!(has(&texts, &scratch_text(b, 2, 0xb2)));
    assert!(!has(&texts, &scratch_text(a, 0, 0xa1)));
    // The CPU releases A's semaphore: A resumes at the word after the acquire.
    r.write(SEM, &[1]);
    r.settle();
    assert_eq!(r.take(), [scratch_text(a, 0, 0xa1)]);
}

#[test]
fn channels_take_turns_of_one_quantum_each() {
    // Each channel has an entry of 2 + 5,000 words: the bind and an immediate per write. The
    // first turn runs a quantum of A (its bind and 4,094 writes) then a quantum of B; the
    // second turn starts after A, so B's rest comes before A's.
    let quantum = usize::try_from(QUANTUM).unwrap();
    let writes = 5_000;
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![r.entry(CMD_A, &immediates(0, 1..=writes))]);
    r.submit(b, vec![r.entry(CMD_FAR, &immediates(1, 1..=writes))]);
    assert_eq!(r.front.turn(), Ok(Turn::Worked));
    assert_eq!(r.front.counters().words, 2 * u64::from(QUANTUM));
    r.settle();
    let first = quantum - 2;
    let rest = usize::from(writes) - first;
    // B's rest follows its first quantum at once, so the two show as one run.
    assert_eq!(
        runs_of_writes(&r.take()),
        [(a.0, first), (b.0, first + rest), (a.0, rest)]
    );
}

#[test]
fn a_channel_that_never_empties_does_not_starve_another() {
    // A gets a new quantum of work before every turn; B's three quanta still finish in three
    // turns.
    let quantum = usize::try_from(QUANTUM).unwrap();
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    let nops = vec![0u32; quantum];
    r.write(CMD_A, &nops);
    let mut long = vec![0u32; 3 * quantum - 1];
    long.extend(write_scratch(1, 0xb1));
    r.submit(b, vec![r.entry(CMD_C, &long)]);
    for _ in 0..4 {
        r.submit(a, vec![raw_entry(CMD_A, quantum, Sync::Proceed)]);
        r.front.turn().unwrap();
    }
    r.settle();
    assert!(has(&r.take(), &scratch_text(b, 1, 0xb1)));
}

#[test]
fn set_reference_reaches_the_ring_with_its_channel() {
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        3,
        m(clc76f::SET_REFERENCE.method()),
        &[0x77],
    )
    .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    assert_eq!(
        r.take(),
        [RingEntry::SetReference {
            channel: a,
            value: 0x77
        }
        .to_string()]
    );
}

#[test]
fn data_words_to_one_trigger_go_out_as_one_run() {
    // Twelve constant-buffer words in one non-incrementing header: one `Run` entry, then the
    // advanced offset, not twelve writes.
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let load = m(clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap());
    let words: Vec<u32> = (1..=12).collect();
    pb.methods(MethodMode::NonIncrementing, 0, load, &words)
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    let run = RingEntry::Run {
        channel: a,
        class: class_3d(),
        method: load,
        words: std::borrow::Cow::Owned(words),
    }
    .to_string();
    let offset = RingEntry::Write {
        channel: a,
        class: class_3d(),
        method: m(clc797::LOAD_CONSTANT_BUFFER_OFFSET.method()),
        value: 12 * 4,
    }
    .to_string();
    let bind = RingEntry::Bind {
        channel: a,
        subchannel: crate::ring::Subchannel::new(0).unwrap(),
        class: class_3d(),
    }
    .to_string();
    assert_eq!(r.take(), [bind, run, offset]);
}

#[test]
fn a_batch_is_written_before_a_host_entry_that_follows_it() {
    // A data run, then a wait for idle on the same channel: the run's entries come first.
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let load = m(clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap());
    pb.methods(MethodMode::NonIncrementing, 0, load, &[1, 2])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 0, m(clc76f::WFI.method()), &[0])
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    let texts = r.take();
    assert_eq!(texts.len(), 4);
    assert!(texts[1].starts_with("run "));
    assert!(texts[2].starts_with("write "));
    assert_eq!(texts[3], format!("wait-for-idle channel={}", a.0));
}

#[test]
fn files_of_other_classes_are_bounded() {
    // One more class than the bound, each bound on subchannel 1 and written once: the oldest
    // file goes, and the 3D file stays.
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    pb.methods(MethodMode::Incrementing, 0, scratch(0), &[1])
        .unwrap();
    let classes: Vec<u32> = (0..=u32::try_from(MAX_OTHER_FILES).unwrap())
        .map(|i| 0x1000 + i)
        .collect();
    for class in &classes {
        pb.methods(
            MethodMode::Incrementing,
            1,
            m(clc76f::SET_OBJECT.method()),
            &[*class],
        )
        .unwrap();
        pb.methods(MethodMode::Incrementing, 1, 0x100, &[*class])
            .unwrap();
    }
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    let files = &r.front.lanes[&a].files;
    assert_eq!(files.len(), MAX_OTHER_FILES + 1);
    assert!(files.file(class_3d()).is_some());
    assert!(files.file(ClassId(classes[0])).is_none());
    assert!(files.file(ClassId(classes[1])).is_some());
    assert!(files.file(ClassId(*classes.last().unwrap())).is_some());
    // Every write was still sent.
    let writes = r.take().iter().filter(|t| t.starts_with("write ")).count();
    assert_eq!(writes, classes.len() + 1);
}

#[test]
fn an_unreadable_entry_stops_the_channel_and_drops_its_queue() {
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut r = rig();
    let a = r.channel(5);
    let b = r.channel(6);
    r.submit(a, vec![raw_entry(unmapped, 1, Sync::Proceed)]);
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa1))]);
    r.submit(b, vec![r.entry(CMD_B, &write_scratch(1, 0xb1))]);
    r.settle();
    let error = ChannelError::ReadFault {
        va: unmapped,
        fault: crate::memory::GpuFault::Unmapped { va: unmapped },
    };
    assert_eq!(r.shared.channel_error(a), Some(error));
    assert_eq!(r.shared.channel_error(b), None);
    assert_eq!(r.front.counters().stopped_channels, 1);
    assert_eq!(r.shared.queued(a), Some(0));
    let texts = r.take();
    assert!(!has(&texts, &scratch_text(a, 0, 0xa1)));
    assert!(has(&texts, &scratch_text(b, 1, 0xb1)));
    // Later submissions leave the queue too, and a barrier is not held by the channel.
    r.submit(a, vec![r.entry(CMD_A, &write_scratch(0, 0xa2))]);
    r.shared.barrier(Ticket(0x51));
    r.settle();
    assert_eq!(r.shared.queued(a), Some(0));
    assert_eq!(
        r.pushed.all(),
        [GpuCompletion::GpuOrdered {
            ticket: Ticket(0x51)
        }]
    );
}

#[test]
fn an_entry_past_the_address_space_stops_the_channel_when_it_starts() {
    let end = crate::channel::GP_ADDRESS_END;
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![raw_entry(end - 4, 2, Sync::Proceed)]);
    r.settle();
    assert_eq!(
        r.shared.channel_error(a),
        Some(ChannelError::EntryPastAddressSpace {
            address: end - 4,
            length: 2
        })
    );
    assert_eq!(r.front.counters().stopped_channels, 1);
}

#[test]
fn a_parked_channel_that_stops_on_its_semaphore_removes_its_waiter() {
    // A parks on SEM; the memory is unmapped; the poll's read faults and stops the channel.
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    acquire(&mut pb, SEM, AcquireOp::Equal, 1);
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    assert!(r.waiters.has_waiter(SPACE, SEM));
    r.mem
        .unmap(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap())
        .unwrap();
    r.settle();
    assert_eq!(
        r.shared.channel_error(a),
        Some(ChannelError::SemaphoreFault {
            address: SEM,
            fault: crate::memory::GpuFault::Unmapped { va: SEM }
        })
    );
    assert!(r.waiters.is_empty());
}

#[test]
fn destroying_a_channel_forgets_everything_it_held() {
    let mut r = rig();
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc797::LOAD_MME_INSTRUCTION_RAM.method()),
        &[0x1234],
    )
    .unwrap();
    acquire(&mut pb, SEM, AcquireOp::Equal, 1);
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    r.settle();
    let space = &r.front.spaces[&SPACE];
    assert_eq!(space.macros.uploads(a, Engine::ThreeD).code_words, 1);
    assert!(r.waiters.has_waiter(SPACE, SEM));
    // A barrier waits for the parked channel; destroying it lets the barrier through.
    r.shared.barrier(Ticket(0x61));
    r.settle();
    assert_eq!(r.pushed.all(), []);
    r.shared.destroy_channel(a);
    r.settle();
    assert!(r.waiters.is_empty());
    assert!(!r.front.lanes.contains_key(&a));
    let space = &r.front.spaces[&SPACE];
    assert_eq!(space.macros.uploads(a, Engine::ThreeD).code_words, 0);
    assert_eq!(r.shared.queued(a), None);
    assert_eq!(
        r.pushed.all(),
        [GpuCompletion::GpuOrdered {
            ticket: Ticket(0x61)
        }]
    );
}

#[test]
fn a_destroyed_channels_stop_error_is_forgotten() {
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![raw_entry(unmapped, 1, Sync::Proceed)]);
    r.settle();
    assert!(r.shared.channel_error(a).is_some());
    r.shared.destroy_channel(a);
    r.settle();
    assert_eq!(r.shared.channel_error(a), None);
}
