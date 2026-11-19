//! Tests for the channel pusher. Streams are built with `fifo::PushBuffer` (slice G1) and run
//! over a `TestGpuMemory` (slice G7); a recording sink keeps everything the channel emits, in
//! order.

use std::sync::Mutex;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_memory::{FlatMemory, PAGE_SIZE, PageRange};

use super::host::{
    CLEAR_FAULTED, FB_FLUSH, ILLEGAL, MEM_OP_A, MEM_OP_B, MEM_OP_C, MEM_OP_D, NON_STALL_INTERRUPT,
    NOP, SEM_ADDR_HI, SEM_ADDR_LO, SEM_EXECUTE, SEM_PAYLOAD_HI, SEM_PAYLOAD_LO, SEMAPHOREA,
    SEMAPHOREB, SEMAPHOREC, SEMAPHORED, SET_OBJECT, SET_REFERENCE, WFI, YIELD,
};
use super::*;
use crate::classes::{clc7c0, clc76f, clc797};
use crate::fifo::{
    Level, PushBuffer, SEC_OP, SEC_OP_RESERVED6, SUBDEVICE_OPCODE, SubdeviceMask, TERT_OP,
};
use crate::memory::{GpuFault, TestGpuMemory};
use crate::ring::{Invalidation, MEM_OP_D_OPERATION, Release, ReleaseOp};
use crate::sema::{
    self, HostSemaphore, PayloadSize, Reduction, ReductionFormat, ReductionOp, ReleaseForm,
};

// ---------------------------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------------------------

/// The CPU memory behind every mapping.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages mapped from `GPU_BASE`.
const PAGES: u64 = 8;
/// Above 32 bits, so the semaphore registers' upper halves matter.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Where command words go.
const CMD: u64 = GPU_BASE;
/// A second command buffer, two pages on.
const CMD2: u64 = GPU_BASE + 2 * PAGE_SIZE;
/// The semaphore the tests acquire on: not page aligned, with distinct bytes in every half.
const SEM: u64 = GPU_BASE + 6 * PAGE_SIZE + 0x10;
/// A channel id other than 0, so a hard-coded id fails.
const ID: ChannelId = ChannelId(7);

fn class_3d() -> ClassId {
    ClassId(clc797::CLASS)
}

fn class_compute() -> ClassId {
    ClassId(clc7c0::CLASS)
}

fn class_i2m() -> ClassId {
    ClassId(cla140::CLASS)
}

fn sub(n: u8) -> Subchannel {
    Subchannel::new(n).unwrap()
}

/// A method number as a header takes it.
fn m(method: u32) -> u16 {
    u16::try_from(method).unwrap()
}

/// Everything the channel emits, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Entry(RingEntry<'static>),
    Method(EngineWord),
    Reference(u32),
    Note(Note),
}

#[derive(Default)]
struct Sink(Vec<Event>);

impl ChannelSink for Sink {
    fn entry(&mut self, entry: RingEntry<'static>) {
        self.0.push(Event::Entry(entry));
    }
    fn method(&mut self, word: EngineWord) {
        self.0.push(Event::Method(word));
    }
    fn set_reference(&mut self, value: u32) {
        self.0.push(Event::Reference(value));
    }
    fn note(&mut self, note: Note) {
        self.0.push(Event::Note(note));
    }
}

struct Rig {
    mem: TestGpuMemory<FlatMemory>,
    channel: Channel,
    sink: Sink,
}

fn make_rig() -> Rig {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    mem.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        CPU_BASE,
    )
    .unwrap();
    Rig {
        mem,
        channel: Channel::new(ID),
        sink: Sink::default(),
    }
}

fn entry_at(address: u64, length: usize) -> GpEntry {
    GpEntry {
        address,
        length: u32::try_from(length).unwrap(),
        ..GpEntry::default()
    }
}

impl Rig {
    fn write(&self, va: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.write(va, &bytes).unwrap();
    }

    fn set_sem(&self, value: u32) {
        self.write(SEM, &[value]);
    }

    /// Writes `words` at `va` and starts the channel on an entry over them.
    fn start(&mut self, va: u64, words: &[u32]) -> GpEntry {
        self.write(va, words);
        let entry = entry_at(va, words.len());
        self.channel.start_entry(entry).unwrap();
        entry
    }

    fn run(&mut self) -> Result<RunOutcome, ChannelError> {
        self.channel.run(&self.mem, &mut self.sink, u32::MAX)
    }

    fn take(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.sink.0)
    }

    fn end_submission(&mut self) -> Result<(), StateError> {
        self.channel.end_submission(&mut self.sink)
    }
}

/// The four `SEMAPHOREA`–`D` words for `address`, `payload` and the operation word.
fn semaphored(pb: &mut PushBuffer, subchannel: u8, address: u64, payload: u32, op: u32) {
    let upper = u32::try_from(address >> 32).unwrap();
    let lower = u32::try_from(address & 0xffff_ffff).unwrap();
    pb.methods(
        MethodMode::Incrementing,
        subchannel,
        m(SEMAPHOREA),
        &[upper, lower, payload, op],
    )
    .unwrap();
}

/// `SEM_ADDR_LO`, `_HI`, `SEM_PAYLOAD_LO`, `_HI` and `SEM_EXECUTE`.
fn sem_execute(pb: &mut PushBuffer, address: u64, payload: u64, op: u32) {
    let hi = u32::try_from(address >> 32).unwrap();
    let lo = u32::try_from(address & 0xffff_ffff).unwrap();
    let payload_lo = u32::try_from(payload & 0xffff_ffff).unwrap();
    let payload_hi = u32::try_from(payload >> 32).unwrap();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEM_ADDR_LO),
        &[lo, hi, payload_lo, payload_hi, op],
    )
    .unwrap();
}

fn acquire_op(op: AcquireOp) -> u32 {
    sema::encode_semaphored(&HostSemaphore::Acquire {
        op,
        size: PayloadSize::Four,
        switch: false,
    })
    .unwrap()
}

fn release_op(timestamp: bool, wfi: bool) -> u32 {
    sema::encode_semaphored(&HostSemaphore::Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp,
        },
        wfi,
    })
    .unwrap()
}

fn method_word(subchannel: u8, class: ClassId, method: u16, value: u32, remaining: u16) -> Event {
    Event::Method(EngineWord {
        subchannel: sub(subchannel),
        class,
        method,
        value,
        remaining,
    })
}

fn dropped(va: u64, word: DroppedWord, first_of_kind: bool) -> Event {
    Event::Note(Note::Dropped {
        va,
        word,
        first_of_kind,
    })
}

/// The GPU address of word `index` of a stream at `base`.
fn word_va(base: u64, index: usize) -> u64 {
    base + 4 * u64::try_from(index).unwrap()
}

fn config() -> Config {
    Config {
        cases: 512,
        rng_seed: RngSeed::Fixed(0x6368_616e),
        failure_persistence: None,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Constants and defaults
// ---------------------------------------------------------------------------------------------

/// The header's byte addresses / 4 and the other facts, written out as literals so a wrong
/// constant cannot agree with itself.
#[test]
fn host_method_numbers_match_clc76f() {
    // `clc76f.h` byte addresses: SET_OBJECT 0x00, ILLEGAL 0x04, NOP 0x08, SEMAPHOREA-D
    // 0x10-0x1c, NON_STALL_INTERRUPT 0x20, FB_FLUSH 0x24, MEM_OP_A-D 0x28-0x34, SET_REFERENCE
    // 0x50, SEM_ADDR_LO 0x5c, SEM_ADDR_HI 0x60, SEM_PAYLOAD_LO 0x64, SEM_PAYLOAD_HI 0x68,
    // SEM_EXECUTE 0x6c, WFI 0x78, YIELD 0x80, CLEAR_FAULTED 0x84.
    let table = [
        (SET_OBJECT, 0x00),
        (ILLEGAL, 0x01),
        (NOP, 0x02),
        (SEMAPHOREA, 0x04),
        (SEMAPHOREB, 0x05),
        (SEMAPHOREC, 0x06),
        (SEMAPHORED, 0x07),
        (NON_STALL_INTERRUPT, 0x08),
        (FB_FLUSH, 0x09),
        (MEM_OP_A, 0x0a),
        (MEM_OP_B, 0x0b),
        (MEM_OP_C, 0x0c),
        (MEM_OP_D, 0x0d),
        (SET_REFERENCE, 0x14),
        (SEM_ADDR_LO, 0x17),
        (SEM_ADDR_HI, 0x18),
        (SEM_PAYLOAD_LO, 0x19),
        (SEM_PAYLOAD_HI, 0x1a),
        (SEM_EXECUTE, 0x1b),
        (WFI, 0x1e),
        (YIELD, 0x20),
        (CLEAR_FAULTED, 0x21),
    ];
    for (constant, literal) in table {
        assert_eq!(constant, literal);
    }
    // Eden-NX2's `BufferMethods::NonPullerMethods`.
    assert_eq!(HOST_METHODS_END, 0x40);
    // Row 49; Eden-NX2's `NvkM2mfSubchannel`; `cla140.h`'s class id.
    assert_eq!(INLINE_TO_MEMORY_SUBCHANNEL, 2);
    assert_eq!(cla140::CLASS, 0xa140);
    // 40-bit GP entry addresses; the 12-bit method field.
    assert_eq!(GP_ADDRESS_END, 0x100_0000_0000);
    assert_eq!(METHOD_MAX, 0xfff);
}

#[test]
fn a_fresh_channel_is_idle_with_zero_registers() {
    let mut rig = make_rig();
    assert_eq!(rig.channel.id(), ID);
    assert_eq!(rig.channel.state(), ChannelState::Idle);
    assert_eq!(rig.channel.reference(), 0);
    for kind in DroppedKind::ALL {
        assert_eq!(rig.channel.dropped_count(kind), 0);
    }
    assert_eq!(rig.run(), Ok(RunOutcome::Idle));
    // Eden's `Puller::Regs regs{}`: a release before any register write is to address 0
    // with payload 0.
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEMAPHORED),
        &[release_op(false, false)],
    )
    .unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [Event::Entry(RingEntry::Release {
            channel: ID,
            release: Release {
                address: 0,
                payload: 0,
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false
                },
                op: ReleaseOp::Write,
            },
        })]
    );
    assert_eq!(rig.channel.state(), ChannelState::Idle);
}

#[test]
fn only_subchannel_2_starts_bound() {
    let mut rig = make_rig();
    for n in 0..u8::try_from(SUBCHANNEL_COUNT).unwrap() {
        let expected = (n == INLINE_TO_MEMORY_SUBCHANNEL).then_some(class_i2m());
        assert_eq!(initial_binding(sub(n)), expected);
        assert_eq!(rig.channel.binding(sub(n)), expected);
    }
    // Eden-NX2 binds 3D, compute, 2D and copy on 0, 1, 3 and 4 as well; here they drop.
    let mut pb = PushBuffer::new();
    for n in 0..5 {
        pb.immediate(n, 0x100, u16::from(n)).unwrap();
    }
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let unbound = |n: u8, first| {
        dropped(
            word_va(CMD, usize::from(n)),
            DroppedWord::UnboundSubchannel {
                subchannel: n,
                method: 0x100,
                value: u32::from(n),
            },
            first,
        )
    };
    assert_eq!(
        rig.take(),
        [
            unbound(0, true),
            unbound(1, false),
            method_word(2, class_i2m(), 0x100, 2, 0),
            unbound(3, false),
            unbound(4, false),
        ]
    );
    assert_eq!(rig.channel.dropped_count(DroppedKind::UnboundSubchannel), 4);
}

// ---------------------------------------------------------------------------------------------
// Acquires: parking and exact-word resume
// ---------------------------------------------------------------------------------------------

#[test]
fn a_parked_channel_resumes_at_the_word_after_the_acquire() {
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x180, 1).unwrap();
    semaphored(&mut pb, 0, SEM, 0x0102, acquire_op(AcquireOp::Equal));
    let after = pb.words().len();
    pb.methods(MethodMode::Incrementing, 2, 0x100, &[0xaaaa, 0xbbbb])
        .unwrap();
    rig.start(CMD, pb.words());

    let acquire = Acquire {
        address: SEM,
        op: AcquireOp::Equal,
        payload: 0x0102,
        method: AcquireMethod::Semaphored,
    };
    assert_eq!(rig.run(), Ok(RunOutcome::Parked(acquire)));
    assert_eq!(rig.channel.state(), ChannelState::Parked(acquire));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x180, 1, 0)]);
    // Parked: run reads nothing, and the channel takes no new entry.
    assert_eq!(rig.run(), Ok(RunOutcome::Parked(acquire)));
    assert_eq!(
        rig.channel.start_entry(entry_at(CMD2, 1)),
        Err(StateError::Parked)
    );

    // The CPU writes the commands after the acquire, then releases it.
    let mut new = PushBuffer::new();
    new.methods(MethodMode::Incrementing, 2, 0x200, &[0x1111, 0x2222])
        .unwrap();
    rig.write(word_va(CMD, after), new.words());
    // A value that differs from the payload only above the low byte is still unmet.
    rig.set_sem(0x0002);
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(false));
    rig.set_sem(0x0102);
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(true));
    assert_eq!(rig.channel.state(), ChannelState::Running);

    // The entry still ends 3 words after the acquire; the resumed run reads the new header
    // and its two data words there, and nothing before the acquire again.
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [
            method_word(2, class_i2m(), 0x200, 0x1111, 1),
            method_word(2, class_i2m(), 0x201, 0x2222, 0),
        ]
    );
}

/// A fault-injecting memory: the first read of `trigger` first writes `patch`, as a CPU that
/// writes the next commands and releases the semaphore between the GPU's reads would.
struct Rewriting<'a> {
    inner: &'a TestGpuMemory<FlatMemory>,
    trigger: u64,
    patch: Mutex<Option<(u64, Vec<u8>)>>,
}

impl GpuMemory for Rewriting<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        if va == self.trigger
            && let Some((at, bytes)) = self.patch.lock().unwrap().take()
        {
            self.inner.write(at, &bytes)?;
        }
        self.inner.read(va, buf)
    }
    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        self.inner.write(va, data)
    }
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
}

#[test]
fn words_written_while_a_met_acquire_is_read_are_read_fresh() {
    let mut rig = make_rig();
    rig.set_sem(5);
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 5, acquire_op(AcquireOp::Equal));
    let after = pb.words().len();
    pb.methods(MethodMode::Incrementing, 2, 0x100, &[0xdead])
        .unwrap();
    rig.write(CMD, pb.words());
    rig.channel
        .start_entry(entry_at(CMD, pb.words().len()))
        .unwrap();

    // The words after the acquire change between the chunk read and the semaphore read.
    let mut new = PushBuffer::new();
    new.methods(MethodMode::Incrementing, 2, 0x300, &[0xbeef])
        .unwrap();
    let bytes = new.words().iter().flat_map(|w| w.to_le_bytes()).collect();
    let memory = Rewriting {
        inner: &rig.mem,
        trigger: SEM,
        patch: Mutex::new(Some((word_va(CMD, after), bytes))),
    };
    assert_eq!(
        rig.channel.run(&memory, &mut rig.sink, u32::MAX),
        Ok(RunOutcome::EntryDone)
    );
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x300, 0xbeef, 0)]);
}

#[test]
fn a_data_word_after_a_parked_acquire_is_read_again() {
    let mut rig = make_rig();
    rig.set_sem(0);
    // Two SEMAPHORED words under one non-incrementing header: acquire, acquire.
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEMAPHOREA),
        &[
            u32::try_from(SEM >> 32).unwrap(),
            u32::try_from(SEM & 0xffff_ffff).unwrap(),
            1,
        ],
    )
    .unwrap();
    pb.methods(
        MethodMode::NonIncrementing,
        0,
        m(SEMAPHORED),
        &[acquire_op(AcquireOp::Equal), acquire_op(AcquireOp::Equal)],
    )
    .unwrap();
    let second = pb.words().len() - 1;
    rig.start(CMD, pb.words());
    assert!(matches!(rig.run(), Ok(RunOutcome::Parked(_))));
    // The CPU turns the second acquire into a release before releasing the first.
    rig.write(word_va(CMD, second), &[release_op(false, false)]);
    rig.set_sem(1);
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(true));
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [Event::Entry(RingEntry::Release {
            channel: ID,
            release: Release {
                address: SEM,
                payload: 1,
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false
                },
                op: ReleaseOp::Write,
            },
        })]
    );
}

#[test]
fn a_latched_waiter_resumes_even_if_the_word_moved_on() {
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 3, acquire_op(AcquireOp::Equal));
    pb.immediate(2, 0x100, 9).unwrap();
    rig.start(CMD, pb.words());
    assert!(matches!(rig.run(), Ok(RunOutcome::Parked(_))));
    // Released to 3 and moved on to 4 before the front looked: the latch resumes it.
    rig.set_sem(4);
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(false));
    assert!(rig.channel.resume_latched());
    assert_eq!(rig.channel.state(), ChannelState::Running);
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 9, 0)]);
    // Not parked: nothing to resume, and a recheck reads nothing.
    assert!(!rig.channel.resume_latched());
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(true));
}

#[test]
fn an_acquire_met_at_once_does_not_park() {
    let mut rig = make_rig();
    rig.set_sem(0x0102);
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 0x0102, acquire_op(AcquireOp::Equal));
    pb.immediate(2, 0x100, 9).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 9, 0)]);
}

#[test]
fn semaphored_acquire_operations_on_both_sides() {
    // (operation, payload, an unmet value, a met value); the values differ above the low byte.
    let cases = [
        (AcquireOp::Equal, 0x0300, 0x0200, 0x0300),
        (AcquireOp::GreaterOrEqual, 0x0300, 0x02ff, 0x0300),
        (AcquireOp::And, 0x0300, 0x0400, 0x0100),
    ];
    for (op, payload, unmet, met) in cases {
        for (value, parks) in [(unmet, true), (met, false)] {
            let mut rig = make_rig();
            rig.set_sem(value);
            let mut pb = PushBuffer::new();
            semaphored(&mut pb, 0, SEM, payload, acquire_op(op));
            rig.start(CMD, pb.words());
            let expected = if parks {
                RunOutcome::Parked(Acquire {
                    address: SEM,
                    op,
                    payload,
                    method: AcquireMethod::Semaphored,
                })
            } else {
                RunOutcome::EntryDone
            };
            assert_eq!(rig.run(), Ok(expected), "{op:?} {value:#x}");
        }
    }
}

#[test]
fn sem_execute_acquire_parks_item_5() {
    // **Guess** (verification item 5): SEM_EXECUTE acquires park, where Eden-NX2 goes on.
    let cases = [
        (AcquireOp::Equal, 0x0300, 0x0200, 0x0300),
        (AcquireOp::GreaterOrEqual, 0x0300, 0x02ff, 0x0300),
        (
            AcquireOp::CircularGreaterOrEqual,
            0x0300,
            0x8000_0300,
            0x8000_02ff,
        ),
        (AcquireOp::And, 0x0300, 0x0400, 0x0100),
        (AcquireOp::Nor, 0x0300, 0x0100, 0x0400),
    ];
    for (op, payload, unmet, met) in cases {
        for (value, parks) in [(unmet, true), (met, false)] {
            let mut rig = make_rig();
            rig.set_sem(value);
            // SEMAPHOREA-C point elsewhere, so reading the wrong registers fails.
            let mut pb = PushBuffer::new();
            pb.methods(
                MethodMode::Incrementing,
                0,
                m(SEMAPHOREA),
                &[0x12, 0x3400_0000, !payload],
            )
            .unwrap();
            let word = sema::encode_sem_execute(&HostSemaphore::Acquire {
                op,
                size: PayloadSize::Four,
                switch: false,
            });
            sem_execute(&mut pb, SEM, u64::from(payload), word);
            rig.start(CMD, pb.words());
            let expected = if parks {
                RunOutcome::Parked(Acquire {
                    address: SEM,
                    op,
                    payload,
                    method: AcquireMethod::SemExecute,
                })
            } else {
                RunOutcome::EntryDone
            };
            assert_eq!(rig.run(), Ok(expected), "{op:?} {value:#x}");
        }
    }
}

#[test]
fn a_64_bit_sem_execute_acquire_compares_the_low_word() {
    // **Guess**: the 32-bit word against SEM_PAYLOAD_LO, as Eden-NX2 and the spec's table.
    let word = sema::encode_sem_execute(&HostSemaphore::Acquire {
        op: AcquireOp::Equal,
        size: PayloadSize::Eight,
        switch: false,
    });
    for (value, parks) in [(0x0000_0005, false), (0x0000_0006, true)] {
        let mut rig = make_rig();
        rig.set_sem(value);
        // The next word holds something other than the high payload word.
        rig.write(SEM + 4, &[0]);
        let mut pb = PushBuffer::new();
        sem_execute(&mut pb, SEM, 0x0000_0007_0000_0005, word);
        rig.start(CMD, pb.words());
        let expected = if parks {
            RunOutcome::Parked(Acquire {
                address: SEM,
                op: AcquireOp::Equal,
                payload: 5,
                method: AcquireMethod::SemExecute,
            })
        } else {
            RunOutcome::EntryDone
        };
        assert_eq!(rig.run(), Ok(expected));
    }
}

#[test]
fn an_unreadable_semaphore_stops_the_channel() {
    // Unreadable at the first check.
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, unmapped, 1, acquire_op(AcquireOp::Equal));
    pb.immediate(2, 0x100, 1).unwrap();
    rig.start(CMD, pb.words());
    let error = ChannelError::SemaphoreFault {
        address: unmapped,
        fault: GpuFault::Unmapped { va: unmapped },
    };
    assert_eq!(rig.run(), Err(error));
    assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
    assert_eq!(rig.take(), []);

    // Unreadable at a recheck: the page is unmapped while the channel is parked.
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 1, acquire_op(AcquireOp::Equal));
    rig.start(CMD, pb.words());
    assert!(matches!(rig.run(), Ok(RunOutcome::Parked(_))));
    rig.mem
        .unmap(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap())
        .unwrap();
    let error = ChannelError::SemaphoreFault {
        address: SEM,
        fault: GpuFault::Unmapped { va: SEM },
    };
    assert_eq!(rig.channel.recheck(&rig.mem), Err(error));
    assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
    // Stopped while parked: the stop wins over the park in every later call.
    assert_eq!(rig.run(), Err(error));
    assert!(!rig.channel.resume_latched());
    assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
    assert_eq!(rig.channel.recheck(&rig.mem), Err(error));
    assert_eq!(rig.take(), []);
}

// ---------------------------------------------------------------------------------------------
// Releases and other host methods
// ---------------------------------------------------------------------------------------------

fn release(address: u64, payload: u64, form: ReleaseForm, op: ReleaseOp) -> Event {
    Event::Entry(RingEntry::Release {
        channel: ID,
        release: Release {
            address,
            payload,
            form,
            op,
        },
    })
}

#[test]
fn semaphored_releases_and_reductions() {
    let mut rig = make_rig();
    let reduction = Reduction {
        op: ReductionOp::Add,
        format: ReductionFormat::Unsigned,
    };
    let reduce = sema::encode_semaphored(&HostSemaphore::Reduce {
        reduction,
        size: PayloadSize::Four,
        wfi: true,
    })
    .unwrap();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 0x1234_5678, release_op(false, false));
    semaphored(&mut pb, 0, SEM, 0x1234_5678, release_op(true, false));
    semaphored(&mut pb, 0, SEM, 0x9abc_def0, reduce);
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    let four = ReleaseForm {
        size: PayloadSize::Four,
        timestamp: false,
    };
    assert_eq!(
        rig.take(),
        [
            release(SEM, 0x1234_5678, four, ReleaseOp::Write),
            release(
                SEM,
                0x1234_5678,
                ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: true
                },
                ReleaseOp::Write
            ),
            release(SEM, 0x9abc_def0, four, ReleaseOp::Reduce(reduction)),
        ]
    );
}

#[test]
fn sem_execute_releases_take_the_payload_width() {
    let mut rig = make_rig();
    let reduction = Reduction {
        op: ReductionOp::Max,
        format: ReductionFormat::Signed,
    };
    let release_word = |size, timestamp| {
        sema::encode_sem_execute(&HostSemaphore::Release {
            form: ReleaseForm { size, timestamp },
            wfi: false,
        })
    };
    let mut pb = PushBuffer::new();
    sem_execute(
        &mut pb,
        SEM,
        0x1111_2222_3333_4444,
        release_word(PayloadSize::Four, false),
    );
    sem_execute(
        &mut pb,
        SEM,
        0x1111_2222_3333_4444,
        release_word(PayloadSize::Eight, true),
    );
    sem_execute(
        &mut pb,
        SEM,
        0x5555_6666_7777_8888,
        sema::encode_sem_execute(&HostSemaphore::Reduce {
            reduction,
            size: PayloadSize::Eight,
            wfi: false,
        }),
    );
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [
            release(
                SEM,
                0x3333_4444,
                ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false
                },
                ReleaseOp::Write
            ),
            release(
                SEM,
                0x1111_2222_3333_4444,
                ReleaseForm {
                    size: PayloadSize::Eight,
                    timestamp: true
                },
                ReleaseOp::Write
            ),
            release(
                SEM,
                0x5555_6666_7777_8888,
                ReleaseForm {
                    size: PayloadSize::Eight,
                    timestamp: false
                },
                ReleaseOp::Reduce(reduction)
            ),
        ]
    );
}

#[test]
fn release_wfi_does_not_change_the_release() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 1, release_op(false, true));
    semaphored(&mut pb, 0, SEM, 1, release_op(false, false));
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let events = rig.take();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0], events[1]);
}

#[test]
fn semaphore_address_upper_bits_follow_the_header() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    // SEMAPHOREA: only 7:0 (0x12) count; Eden-NX2 would shift all 32 bits. SEMAPHOREB: bits
    // 1:0 are not part of the address.
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEMAPHOREA),
        &[0xffff_ff12, 0x3456_789b, 7, release_op(false, false)],
    )
    .unwrap();
    // SEM_ADDR_LO 0x0000_1003, SEM_ADDR_HI 0xffff_ff34: Eden-NX2 would keep 25 bits.
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEM_ADDR_LO),
        &[
            0x0000_1003,
            0xffff_ff34,
            8,
            0,
            sema::encode_sem_execute(&HostSemaphore::Release {
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false,
                },
                wfi: false,
            }),
        ],
    )
    .unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let four = ReleaseForm {
        size: PayloadSize::Four,
        timestamp: false,
    };
    assert_eq!(
        rig.take(),
        [
            release(0x12_3456_7898, 7, four, ReleaseOp::Write),
            release(0x34_0000_1000, 8, four, ReleaseOp::Write),
        ]
    );
}

#[test]
fn host_methods_run_on_any_subchannel() {
    // Subchannel 5 has nothing bound; host methods do not look.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 5, SEM, 1, release_op(false, false));
    pb.immediate(6, m(WFI), 0).unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    assert_eq!(
        rig.take(),
        [
            release(
                SEM,
                1,
                ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false
                },
                ReleaseOp::Write
            ),
            Event::Entry(RingEntry::WaitForIdle { channel: ID }),
        ]
    );
}

#[test]
fn mem_op_d_triggers_the_operation() {
    let every = [
        Invalidation::Membar,
        Invalidation::MmuTlbInvalidate,
        Invalidation::MmuTlbInvalidateTargeted,
        Invalidation::MmuOperation,
        Invalidation::L2PeermemInvalidate,
        Invalidation::L2SysmemInvalidate,
        Invalidation::L2CleanComptags,
        Invalidation::L2FlushDirty,
        Invalidation::L2WaitForSysPendingReads,
        Invalidation::AccessCounterClear,
    ];
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    // MEM_OP_A-C first: they emit nothing (Eden-NX2 invalidates on MEM_OP_B).
    pb.methods(MethodMode::Incrementing, 0, m(MEM_OP_A), &[1, 2, 3])
        .unwrap();
    for what in every {
        let word = MEM_OP_D_OPERATION.insert(0x1234, what.operation()).unwrap();
        pb.methods(MethodMode::Incrementing, 0, m(MEM_OP_D), &[word])
            .unwrap();
    }
    // Operation 0x1f is not defined.
    pb.methods(MethodMode::Incrementing, 0, m(MEM_OP_D), &[0xf800_0000])
        .unwrap();
    let last = pb.words().len() - 1;
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let mut expected: Vec<Event> = every
        .iter()
        .map(|&what| Event::Entry(RingEntry::Invalidate { channel: ID, what }))
        .collect();
    expected.push(dropped(
        word_va(CMD, last),
        DroppedWord::UndefinedMemOp { operation: 0x1f },
        true,
    ));
    assert_eq!(rig.take(), expected);
}

#[test]
fn set_reference_and_wfi() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SET_REFERENCE),
        &[0x0102_0304],
    )
    .unwrap();
    pb.methods(MethodMode::Incrementing, 3, m(WFI), &[1])
        .unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    assert_eq!(
        rig.take(),
        [
            Event::Reference(0x0102_0304),
            Event::Entry(RingEntry::WaitForIdle { channel: ID }),
        ]
    );
    assert_eq!(rig.channel.reference(), 0x0102_0304);
}

#[test]
fn no_op_host_methods_emit_nothing() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    for method in [
        ILLEGAL,
        NOP,
        NON_STALL_INTERRUPT,
        FB_FLUSH,
        MEM_OP_A,
        MEM_OP_B,
        MEM_OP_C,
        YIELD,
        CLEAR_FAULTED,
        SEMAPHOREA,
        SEMAPHOREB,
        SEMAPHOREC,
        SEM_ADDR_LO,
        SEM_ADDR_HI,
        SEM_PAYLOAD_LO,
        SEM_PAYLOAD_HI,
    ] {
        pb.methods(MethodMode::Incrementing, 0, m(method), &[0xffff_ffff])
            .unwrap();
    }
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), []);
}

#[test]
fn unknown_host_methods_are_dropped() {
    let known = [
        SET_OBJECT,
        ILLEGAL,
        NOP,
        SEMAPHOREA,
        SEMAPHOREB,
        SEMAPHOREC,
        SEMAPHORED,
        NON_STALL_INTERRUPT,
        FB_FLUSH,
        MEM_OP_A,
        MEM_OP_B,
        MEM_OP_C,
        MEM_OP_D,
        SET_REFERENCE,
        SEM_ADDR_LO,
        SEM_ADDR_HI,
        SEM_PAYLOAD_LO,
        SEM_PAYLOAD_HI,
        SEM_EXECUTE,
        WFI,
        YIELD,
        CLEAR_FAULTED,
    ];
    // Every method of the generated table below 0x40 is in the known set.
    for method in clc76f::HEADER.methods {
        if method.method() < HOST_METHODS_END {
            assert!(known.contains(&method.method()), "{}", method.name);
        }
    }
    let unknown: Vec<u32> = (0..HOST_METHODS_END)
        .filter(|method| !known.contains(method))
        .collect();
    // Eden-NX2's Maxwell syncpoint methods are among them.
    assert!(unknown.contains(&0x1c) && unknown.contains(&0x1d));
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    for &method in &unknown {
        pb.methods(MethodMode::Incrementing, 1, m(method), &[0x100 + method])
            .unwrap();
    }
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let expected: Vec<Event> = unknown
        .iter()
        .enumerate()
        .map(|(i, &method)| {
            dropped(
                word_va(CMD, 2 * i + 1),
                DroppedWord::UnknownHostMethod {
                    method: m(method),
                    value: 0x100 + method,
                },
                i == 0,
            )
        })
        .collect();
    assert_eq!(rig.take(), expected);
    assert_eq!(
        rig.channel.dropped_count(DroppedKind::UnknownHostMethod),
        u64::try_from(unknown.len()).unwrap()
    );
}

// ---------------------------------------------------------------------------------------------
// Binds and engine methods
// ---------------------------------------------------------------------------------------------

#[test]
fn engine_methods_follow_their_mode() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::Incrementing, 1, m(SET_OBJECT), &[clc797::CLASS])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 1, 0x200, &[10, 11, 12])
        .unwrap();
    pb.methods(MethodMode::NonIncrementing, 1, 0x300, &[20, 21])
        .unwrap();
    pb.methods(MethodMode::IncrementOnce, 1, 0x400, &[30, 31, 32])
        .unwrap();
    pb.immediate(1, 0x500, 0x1fff).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    let c = class_3d();
    assert_eq!(
        rig.take(),
        [
            Event::Entry(RingEntry::Bind {
                channel: ID,
                subchannel: sub(1),
                class: c
            }),
            method_word(1, c, 0x200, 10, 2),
            method_word(1, c, 0x201, 11, 1),
            method_word(1, c, 0x202, 12, 0),
            method_word(1, c, 0x300, 20, 1),
            method_word(1, c, 0x300, 21, 0),
            method_word(1, c, 0x400, 30, 2),
            method_word(1, c, 0x401, 31, 1),
            method_word(1, c, 0x401, 32, 0),
            method_word(1, c, 0x500, 0x1fff, 0),
        ]
    );
    assert_eq!(rig.channel.binding(sub(1)), Some(c));
}

#[test]
fn a_method_run_crosses_from_host_to_engine_methods() {
    // An incrementing run from 0x3f goes to the host (unknown) then to the engine.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::Incrementing, 2, 0x3f, &[1, 2])
        .unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    assert_eq!(
        rig.take(),
        [
            dropped(
                word_va(CMD, 1),
                DroppedWord::UnknownHostMethod {
                    method: 0x3f,
                    value: 1
                },
                true
            ),
            method_word(2, class_i2m(), 0x40, 2, 0),
        ]
    );
}

#[test]
fn class_zero_unbinds_the_subchannel() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(1, m(SET_OBJECT), 0).unwrap();
    pb.methods(MethodMode::Incrementing, 3, m(SET_OBJECT), &[clc7c0::CLASS])
        .unwrap();
    pb.immediate(3, 0x100, 1).unwrap();
    pb.immediate(3, m(SET_OBJECT), 0).unwrap();
    pb.immediate(3, 0x100, 2).unwrap();
    // Subchannel 2's starting binding goes too.
    pb.immediate(2, m(SET_OBJECT), 0).unwrap();
    pb.immediate(2, 0x100, 3).unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    let bind = |n, class| {
        Event::Entry(RingEntry::Bind {
            channel: ID,
            subchannel: sub(n),
            class,
        })
    };
    assert_eq!(
        rig.take(),
        [
            bind(1, ClassId(0)),
            bind(3, class_compute()),
            method_word(3, class_compute(), 0x100, 1, 0),
            bind(3, ClassId(0)),
            dropped(
                word_va(CMD, 5),
                DroppedWord::UnboundSubchannel {
                    subchannel: 3,
                    method: 0x100,
                    value: 2
                },
                true
            ),
            bind(2, ClassId(0)),
            dropped(
                word_va(CMD, 7),
                DroppedWord::UnboundSubchannel {
                    subchannel: 2,
                    method: 0x100,
                    value: 3
                },
                false
            ),
        ]
    );
    assert_eq!(rig.channel.binding(sub(3)), None);
    assert_eq!(rig.channel.binding(sub(2)), None);
}

#[test]
fn bind_reads_nvclass_only() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    // Bits above 20 set (Eden-NX2 would keep the whole word), and the ENGINE field set to SW.
    let engine_sw = clc76f::SET_OBJECT_ENGINE
        .insert(clc7c0::CLASS, clc76f::SET_OBJECT_ENGINE_SW)
        .unwrap();
    pb.methods(
        MethodMode::Incrementing,
        4,
        m(SET_OBJECT),
        &[0xffe0_0000 | clc797::CLASS],
    )
    .unwrap();
    pb.methods(MethodMode::Incrementing, 5, m(SET_OBJECT), &[engine_sw])
        .unwrap();
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    assert_eq!(rig.channel.binding(sub(4)), Some(class_3d()));
    assert_eq!(rig.channel.binding(sub(5)), Some(class_compute()));
}

#[test]
fn old_format_headers_run_as_methods() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    // The all-zero NOP word: an old-format header with a count of 0.
    pb.data(&[0]);
    pb.header(&MethodHeader::Legacy {
        mode: MethodMode::Incrementing,
        subchannel: 2,
        method: 0x100,
        count: 2,
    })
    .unwrap()
    .data(&[1, 2]);
    pb.header(&MethodHeader::Legacy {
        mode: MethodMode::NonIncrementing,
        subchannel: 2,
        method: 0x200,
        count: 2,
    })
    .unwrap()
    .data(&[3, 4]);
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    let c = class_i2m();
    assert_eq!(
        rig.take(),
        [
            method_word(2, c, 0x100, 1, 1),
            method_word(2, c, 0x101, 2, 0),
            method_word(2, c, 0x200, 3, 1),
            method_word(2, c, 0x200, 4, 0),
        ]
    );
}

#[test]
fn subdevice_mask_operations_are_skipped() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.header(&MethodHeader::SubdeviceMask(SubdeviceMask::Set {
        mask: 0xfff,
    }))
    .unwrap();
    pb.header(&MethodHeader::SubdeviceMask(SubdeviceMask::Store {
        mask: 0x123,
    }))
    .unwrap();
    pb.header(&MethodHeader::SubdeviceMask(SubdeviceMask::Use))
        .unwrap();
    pb.immediate(2, 0x100, 1).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 1, 0)]);
}

#[test]
fn end_segment_ends_the_entry_guess_item_9() {
    // **Guess** (verification item 9): opcode 7 ends the entry; Eden-NX2 skips the word.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    pb.header(&MethodHeader::EndSegment).unwrap();
    pb.immediate(2, 0x100, 2).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.channel.state(), ChannelState::Idle);
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 1, 0)]);
    // The next entry runs normally.
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 3).unwrap();
    rig.start(CMD2, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 3, 0)]);
}

// ---------------------------------------------------------------------------------------------
// Split methods
// ---------------------------------------------------------------------------------------------

#[test]
fn a_method_split_across_entries_continues_in_the_next_entry() {
    let mut rig = make_rig();
    let mut first = PushBuffer::new();
    first
        .header(&MethodHeader::Methods {
            mode: MethodMode::Incrementing,
            subchannel: 2,
            method: 0x100,
            count: 3,
        })
        .unwrap()
        .data(&[10]);
    rig.start(CMD, first.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    // The rest arrives in a subroutine entry elsewhere (row 73).
    let mut second = PushBuffer::new();
    second.data(&[11, 12]);
    second.immediate(2, 0x200, 5).unwrap();
    rig.write(CMD2, second.words());
    rig.channel
        .start_entry(GpEntry {
            level: Level::Subroutine,
            ..entry_at(CMD2, second.words().len())
        })
        .unwrap();
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    let c = class_i2m();
    assert_eq!(
        rig.take(),
        [
            method_word(2, c, 0x100, 10, 2),
            method_word(2, c, 0x101, 11, 1),
            method_word(2, c, 0x102, 12, 0),
            method_word(2, c, 0x200, 5, 0),
        ]
    );
}

#[test]
fn a_method_split_across_submissions_is_noted_once() {
    let mut rig = make_rig();
    let carried = |method, remaining| {
        Event::Note(Note::MethodCarried {
            subchannel: sub(2),
            method,
            remaining,
        })
    };
    // Submission 1: the header and one of four words.
    let mut pb = PushBuffer::new();
    pb.header(&MethodHeader::Methods {
        mode: MethodMode::Incrementing,
        subchannel: 2,
        method: 0x100,
        count: 4,
    })
    .unwrap()
    .data(&[1]);
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    rig.end_submission().unwrap();
    // Submission 2: one more word; still pending, not noted again.
    rig.start(CMD2, &[2]);
    rig.run().unwrap();
    rig.end_submission().unwrap();
    // Submission 3: the last two words, then a new method left pending, which is noted.
    let mut pb = PushBuffer::new();
    pb.data(&[3, 4]);
    pb.header(&MethodHeader::Methods {
        mode: MethodMode::NonIncrementing,
        subchannel: 2,
        method: 0x300,
        count: 2,
    })
    .unwrap()
    .data(&[5]);
    rig.start(CMD, pb.words());
    rig.run().unwrap();
    rig.end_submission().unwrap();
    // A submission with nothing pending notes nothing.
    rig.start(CMD2, &[6]);
    rig.run().unwrap();
    rig.end_submission().unwrap();
    let c = class_i2m();
    assert_eq!(
        rig.take(),
        [
            method_word(2, c, 0x100, 1, 3),
            carried(0x101, 3),
            method_word(2, c, 0x101, 2, 2),
            method_word(2, c, 0x102, 3, 1),
            method_word(2, c, 0x103, 4, 0),
            method_word(2, c, 0x300, 5, 1),
            carried(0x300, 1),
            method_word(2, c, 0x300, 6, 0),
        ]
    );
}

#[test]
fn end_submission_needs_an_idle_channel() {
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    pb.immediate(2, 0x100, 2).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(
        rig.channel.run(&rig.mem, &mut rig.sink, 1),
        Ok(RunOutcome::QuantumUsed)
    );
    assert_eq!(rig.end_submission(), Err(StateError::Running));
    assert_eq!(
        rig.channel.start_entry(entry_at(CMD2, 1)),
        Err(StateError::Running)
    );
    rig.run().unwrap();
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 1, acquire_op(AcquireOp::Equal));
    rig.start(CMD, pb.words());
    assert!(matches!(rig.run(), Ok(RunOutcome::Parked(_))));
    assert_eq!(rig.end_submission(), Err(StateError::Parked));
}

// ---------------------------------------------------------------------------------------------
// Quantum and reading
// ---------------------------------------------------------------------------------------------

#[test]
fn the_quantum_bounds_the_words_read_per_call() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    for value in 0..5 {
        pb.immediate(2, 0x100, value).unwrap();
    }
    rig.start(CMD, pb.words());
    let mut run = |quantum| rig.channel.run(&rig.mem, &mut rig.sink, quantum);
    assert_eq!(run(0), Ok(RunOutcome::QuantumUsed));
    assert_eq!(run(2), Ok(RunOutcome::QuantumUsed));
    assert_eq!(run(2), Ok(RunOutcome::QuantumUsed));
    // One word left: read, then the end is found before the quantum.
    assert_eq!(run(2), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take().len(), 5);

    // An entry whose last word is the quantum's last ends in that call.
    let mut pb = PushBuffer::new();
    for value in 0..4 {
        pb.immediate(2, 0x100, value).unwrap();
    }
    rig.start(CMD, pb.words());
    let mut run = |quantum| rig.channel.run(&rig.mem, &mut rig.sink, quantum);
    assert_eq!(run(2), Ok(RunOutcome::QuantumUsed));
    assert_eq!(run(2), Ok(RunOutcome::EntryDone));

    // A zero-length entry ends at once.
    rig.channel.start_entry(entry_at(CMD, 0)).unwrap();
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
}

#[test]
fn words_read_ahead_are_not_kept_across_run_calls() {
    // Rewritten between two quanta: the second call reads the new word, not the copy read
    // ahead by the first call.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    pb.immediate(2, 0x100, 2).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(
        rig.channel.run(&rig.mem, &mut rig.sink, 1),
        Ok(RunOutcome::QuantumUsed)
    );
    let mut new = PushBuffer::new();
    new.immediate(2, 0x300, 3).unwrap();
    rig.write(word_va(CMD, 1), new.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [
            method_word(2, class_i2m(), 0x100, 1, 0),
            method_word(2, class_i2m(), 0x300, 3, 0),
        ]
    );

    // Unmapped between two quanta: the second call faults on the word it needs.
    let mut rig = make_rig();
    rig.start(CMD, pb.words());
    assert_eq!(
        rig.channel.run(&rig.mem, &mut rig.sink, 1),
        Ok(RunOutcome::QuantumUsed)
    );
    rig.mem
        .unmap(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap())
        .unwrap();
    let va = word_va(CMD, 1);
    let error = ChannelError::ReadFault {
        va,
        fault: GpuFault::Unmapped { va },
    };
    assert_eq!(rig.run(), Err(error));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 1, 0)]);
}

#[test]
fn an_entry_across_a_page_boundary_reads_both_pages() {
    let mut rig = make_rig();
    let start = GPU_BASE + PAGE_SIZE - 8;
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::NonIncrementing, 2, 0x100, &[1, 2, 3, 4])
        .unwrap();
    rig.start(start, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    let c = class_i2m();
    assert_eq!(
        rig.take(),
        [
            method_word(2, c, 0x100, 1, 3),
            method_word(2, c, 0x100, 2, 2),
            method_word(2, c, 0x100, 3, 1),
            method_word(2, c, 0x100, 4, 0),
        ]
    );
}

#[test]
fn an_unmapped_second_page_is_not_read_before_it_is_needed() {
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    semaphored(&mut pb, 0, SEM, 1, acquire_op(AcquireOp::Equal));
    let len = pb.words().len();
    // The stream ends exactly at the end of the last mapped page; the entry runs on.
    let end = GPU_BASE + PAGES * PAGE_SIZE;
    let start = end - 4 * u64::try_from(len).unwrap();
    rig.write(start, pb.words());
    rig.channel.start_entry(entry_at(start, len + 3)).unwrap();
    assert!(matches!(rig.run(), Ok(RunOutcome::Parked(_))));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 1, 0)]);
    rig.set_sem(1);
    assert_eq!(rig.channel.recheck(&rig.mem), Ok(true));
    let error = ChannelError::ReadFault {
        va: end,
        fault: GpuFault::Unmapped { va: end },
    };
    assert_eq!(rig.run(), Err(error));
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[test]
fn an_unmapped_entry_stops_the_channel_for_good() {
    let mut rig = make_rig();
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    rig.channel.start_entry(entry_at(unmapped, 4)).unwrap();
    let error = ChannelError::ReadFault {
        va: unmapped,
        fault: GpuFault::Unmapped { va: unmapped },
    };
    assert_eq!(rig.run(), Err(error));
    // Sticky: every later call reports the same error.
    assert_eq!(rig.run(), Err(error));
    assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
    assert_eq!(
        rig.channel.start_entry(entry_at(CMD, 1)),
        Err(StateError::Stopped(error))
    );
    assert_eq!(rig.channel.recheck(&rig.mem), Err(error));
    assert_eq!(rig.end_submission(), Err(StateError::Stopped(error)));
    assert!(!rig.channel.resume_latched());
    assert_eq!(rig.take(), []);
}

#[test]
fn an_entry_past_the_address_space_stops_the_channel() {
    // Ends exactly at 2^40: accepted, and then faults reading (nothing is mapped there).
    let mut rig = make_rig();
    let at_end = entry_at(GP_ADDRESS_END - 4 * 3, 3);
    assert_eq!(rig.channel.start_entry(at_end), Ok(()));
    assert!(matches!(rig.run(), Err(ChannelError::ReadFault { .. })));
    // One word further: refused before anything is read.
    let mut rig = make_rig();
    let past = entry_at(GP_ADDRESS_END - 4 * 3 + 4, 3);
    let error = ChannelError::EntryPastAddressSpace {
        address: GP_ADDRESS_END - 8,
        length: 3,
    };
    assert_eq!(
        rig.channel.start_entry(past),
        Err(StateError::Stopped(error))
    );
    assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
    assert_eq!(rig.run(), Err(error));
}

#[test]
fn an_unaligned_entry_stops_the_channel() {
    // Every misalignment is refused before anything is read.
    for misalign in 1..WORD_BYTES {
        let mut rig = make_rig();
        let address = CMD + misalign;
        let error = ChannelError::UnalignedEntry { address };
        assert_eq!(
            rig.channel.start_entry(entry_at(address, 2)),
            Err(StateError::Stopped(error))
        );
        assert_eq!(rig.channel.state(), ChannelState::Stopped(error));
        assert_eq!(rig.run(), Err(error));
        assert_eq!(rig.take(), []);
    }
    // The aligned neighbour is accepted and runs.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    rig.write(CMD + WORD_BYTES, pb.words());
    assert_eq!(
        rig.channel.start_entry(entry_at(CMD + WORD_BYTES, 1)),
        Ok(())
    );
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.take(), [method_word(2, class_i2m(), 0x100, 1, 0)]);

    // Both faults at once: unaligned, and past the address space. Alignment is checked first.
    let mut rig = make_rig();
    let address = GP_ADDRESS_END - 2;
    assert_eq!(
        rig.channel.start_entry(entry_at(address, 2)),
        Err(StateError::Stopped(ChannelError::UnalignedEntry {
            address
        }))
    );
}

#[test]
fn a_busy_channel_refuses_before_the_entry_is_checked() {
    let past = entry_at(GP_ADDRESS_END - 4, 2);
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    pb.immediate(2, 0x100, 2).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(
        rig.channel.run(&rig.mem, &mut rig.sink, 1),
        Ok(RunOutcome::QuantumUsed)
    );
    // Both faults at once: running, and an entry past the address space (or unaligned).
    assert_eq!(rig.channel.start_entry(past), Err(StateError::Running));
    assert_eq!(
        rig.channel.start_entry(entry_at(CMD + 2, 1)),
        Err(StateError::Running)
    );
    assert_eq!(rig.channel.state(), ChannelState::Running);
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
}

#[test]
fn a_method_run_past_0xfff_stops_the_channel() {
    let max = m(METHOD_MAX);
    // (mode, method, count, past): the last method reached is exactly 0xfff, or one past it.
    let cases = [
        (MethodMode::Incrementing, max - 1, 2, false),
        (MethodMode::Incrementing, max - 1, 3, true),
        (MethodMode::Incrementing, max, 0, false),
        (MethodMode::IncrementOnce, max, 1, false),
        (MethodMode::IncrementOnce, max, 2, true),
        (MethodMode::IncrementOnce, max - 1, 5, false),
        (MethodMode::NonIncrementing, max, 0x1fff, false),
    ];
    for (mode, method, count, past) in cases {
        assert_eq!(
            runs_past_method_space(mode, method, count),
            past,
            "{mode:?} {method:#x} {count}"
        );
    }
    // Through a channel: the accepted header runs; the refused one stops before its data.
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::Incrementing, 2, max - 1, &[1, 2])
        .unwrap();
    pb.methods(MethodMode::Incrementing, 2, max - 1, &[3, 4, 5])
        .unwrap();
    rig.start(CMD, pb.words());
    let error = ChannelError::MethodPastSpace {
        method: max - 1,
        count: 3,
    };
    assert_eq!(rig.run(), Err(error));
    let c = class_i2m();
    assert_eq!(
        rig.take(),
        [
            method_word(2, c, max - 1, 1, 1),
            method_word(2, c, max, 2, 0)
        ]
    );
}

#[test]
fn malformed_words_are_dropped_and_the_stream_goes_on() {
    let reserved = SEC_OP.insert(0x1234, SEC_OP_RESERVED6).unwrap();
    let bad_subdevice = SUBDEVICE_OPCODE
        .insert(TERT_OP.insert(0, 1).unwrap(), 0x7)
        .unwrap();
    let bad_group2 = TERT_OP
        .insert(
            SEC_OP.insert(0, crate::fifo::SEC_OP_GRP2_USE_TERT).unwrap(),
            1,
        )
        .unwrap();
    let bad_sem_execute = 7;
    let bad_semaphored = 3;
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.data(&[reserved, bad_subdevice, bad_group2]);
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEM_EXECUTE),
        &[bad_sem_execute],
    )
    .unwrap();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(SEMAPHORED),
        &[bad_semaphored],
    )
    .unwrap();
    pb.immediate(2, 0x100, 1).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(
        rig.take(),
        [
            dropped(
                word_va(CMD, 0),
                DroppedWord::BadHeader(fifo::DecodeError::ReservedOpcode { word: reserved }),
                true
            ),
            dropped(
                word_va(CMD, 1),
                DroppedWord::BadHeader(fifo::DecodeError::UndefinedSubdeviceOp {
                    word: bad_subdevice
                }),
                false
            ),
            dropped(
                word_va(CMD, 2),
                DroppedWord::BadHeader(fifo::DecodeError::UndefinedGroup2Op { word: bad_group2 }),
                false
            ),
            dropped(
                word_va(CMD, 4),
                DroppedWord::BadSemaphore(sema::DecodeError::UndefinedOperation {
                    word: sema::SEM_EXECUTE,
                    value: 7
                }),
                true
            ),
            dropped(
                word_va(CMD, 6),
                DroppedWord::BadSemaphore(sema::DecodeError::UndefinedOperation {
                    word: sema::SEMAPHORED,
                    value: 3
                }),
                false
            ),
            method_word(2, class_i2m(), 0x100, 1, 0),
        ]
    );
    assert_eq!(rig.channel.dropped_count(DroppedKind::BadHeader), 3);
    assert_eq!(rig.channel.dropped_count(DroppedKind::BadSemaphore), 2);
    assert_eq!(rig.channel.dropped_count(DroppedKind::UnknownHostMethod), 0);
}

#[test]
fn every_dropped_word_has_its_kind() {
    let words = [
        DroppedWord::BadHeader(fifo::DecodeError::ReservedOpcode { word: 1 }),
        DroppedWord::UnknownHostMethod {
            method: 3,
            value: 4,
        },
        DroppedWord::UnboundSubchannel {
            subchannel: 1,
            method: 0x100,
            value: 2,
        },
        DroppedWord::BadSemaphore(sema::DecodeError::UndefinedOperation {
            word: sema::SEMAPHORED,
            value: 3,
        }),
        DroppedWord::UndefinedMemOp { operation: 0 },
    ];
    for (i, (word, kind)) in words.iter().zip(DroppedKind::ALL).enumerate() {
        assert_eq!(word.kind(), kind);
        assert_eq!(kind.index(), i);
    }
}

#[test]
fn errors_read_as_plain_words() {
    let fault = GpuFault::Unmapped { va: 0x1000 };
    assert_eq!(
        ChannelError::ReadFault { va: 0x1000, fault }.to_string(),
        "the command word at GPU address 0x1000 could not be read: GPU address 0x1000 is not mapped"
    );
    assert_eq!(
        ChannelError::SemaphoreFault {
            address: 0x1000,
            fault
        }
        .to_string(),
        "the semaphore at GPU address 0x1000 could not be read: GPU address 0x1000 is not mapped"
    );
    assert_eq!(
        ChannelError::EntryPastAddressSpace {
            address: 0xff_ffff_fff8,
            length: 3
        }
        .to_string(),
        "the GP entry of 3 words at GPU address 0xfffffffff8 runs past the 40-bit address space"
    );
    assert_eq!(
        ChannelError::MethodPastSpace {
            method: 0xffe,
            count: 3
        }
        .to_string(),
        "a method header for 3 words from method 0xffe runs past method 0xfff"
    );
    assert_eq!(
        ChannelError::UnalignedEntry {
            address: 0x12_3400_0002
        }
        .to_string(),
        "the GP entry at GPU address 0x1234000002 is not 4-byte aligned"
    );
    assert_eq!(
        DroppedWord::BadHeader(fifo::DecodeError::ReservedOpcode { word: 0xc000_0000 }).to_string(),
        "method header 0xc0000000 uses the reserved opcode 6"
    );
    assert_eq!(
        DroppedWord::UnknownHostMethod {
            method: 0x1c,
            value: 5
        }
        .to_string(),
        "host method 0x1c is not defined (data 0x00000005)"
    );
    assert_eq!(
        DroppedWord::UnboundSubchannel {
            subchannel: 3,
            method: 0x100,
            value: 0
        }
        .to_string(),
        "method 0x100 was sent on subchannel 3, which has no class bound"
    );
    assert_eq!(
        DroppedWord::BadSemaphore(sema::DecodeError::UndefinedOperation {
            word: sema::SEM_EXECUTE,
            value: 7
        })
        .to_string(),
        "SEM_EXECUTE operation 0x7 is not defined"
    );
    assert_eq!(
        DroppedWord::UndefinedMemOp { operation: 0 }.to_string(),
        "MEM_OP_D operation 0x0 is not defined"
    );
    assert_eq!(
        StateError::Running.to_string(),
        "the channel is still running a GP entry"
    );
    assert_eq!(
        StateError::Parked.to_string(),
        "the channel is parked on an acquire"
    );
    assert_eq!(
        StateError::Stopped(ChannelError::MethodPastSpace {
            method: 0xffe,
            count: 3
        })
        .to_string(),
        "the channel has stopped: a method header for 3 words from method 0xffe runs past method 0xfff"
    );
}

// ---------------------------------------------------------------------------------------------
// Property: no word sequence panics
// ---------------------------------------------------------------------------------------------

/// Words biased toward meaningful ones: headers for host methods and bound subchannels, and
/// semaphore operation words, as well as anything at all.
fn any_word() -> impl Strategy<Value = u32> {
    prop_oneof![
        any::<u32>(),
        (0u32..8, 0u32..0x48, 0u32..8, 0u32..8).prop_map(|(op, method, subchannel, count)| {
            let word = SEC_OP.insert(0, op).unwrap();
            let word = crate::fifo::METHOD_ADDRESS.insert(word, method).unwrap();
            let word = crate::fifo::METHOD_SUBCHANNEL
                .insert(word, subchannel)
                .unwrap();
            crate::fifo::METHOD_COUNT.insert(word, count).unwrap()
        }),
        0u32..0x20,
        Just(0),
    ]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn no_word_sequence_panics(
        words in prop::collection::vec(any_word(), 0..256),
        cuts in prop::collection::vec(0usize..256, 0..6),
        sem in any::<u32>(),
        quantum in 1u32..64,
    ) {
        let mut rig = make_rig();
        rig.set_sem(sem);
        rig.write(CMD, &words);
        // Split the words into entries at the cut points.
        let mut bounds: Vec<usize> = cuts.iter().map(|&c| c.min(words.len())).collect();
        bounds.push(0);
        bounds.push(words.len());
        bounds.sort_unstable();
        bounds.dedup();
        let entries: Vec<GpEntry> = bounds
            .windows(2)
            .map(|w| entry_at(word_va(CMD, w[0]), w[1] - w[0]))
            .collect();
        // Every call reads a word, ends an entry or resumes a park, so this bounds the run.
        let mut budget = 4 * (words.len() + entries.len()) + 8;
        let mut next = entries.iter();
        loop {
            prop_assert!(budget > 0, "the channel made no progress");
            budget -= 1;
            match rig.channel.state() {
                ChannelState::Stopped(_) => break,
                ChannelState::Parked(_) => {
                    prop_assert!(rig.channel.resume_latched());
                }
                ChannelState::Idle => {
                    let Some(&entry) = next.next() else { break };
                    prop_assert_eq!(rig.channel.start_entry(entry), Ok(()));
                }
                ChannelState::Running => {
                    let outcome = rig.channel.run(&rig.mem, &mut rig.sink, quantum);
                    prop_assert!(outcome != Ok(RunOutcome::Idle));
                }
            }
        }
        // Nothing past the 12-bit method space reaches the sink.
        for event in &rig.sink.0 {
            if let Event::Method(word) = event {
                prop_assert!(u32::from(word.method) <= METHOD_MAX);
                prop_assert!(u32::from(word.method) >= HOST_METHODS_END);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Holding (the macro host's catch-up, slice G13)
// ---------------------------------------------------------------------------------------------

/// A sink that starts holding the channel once it has taken `hold_after` method words.
struct HoldingSink {
    inner: Sink,
    hold_after: usize,
    taken: usize,
    held: bool,
}

impl HoldingSink {
    fn new(hold_after: usize) -> Self {
        Self {
            inner: Sink::default(),
            hold_after,
            taken: 0,
            held: false,
        }
    }
}

impl ChannelSink for HoldingSink {
    fn entry(&mut self, entry: RingEntry<'static>) {
        self.inner.entry(entry);
    }
    fn method(&mut self, word: EngineWord) {
        self.inner.method(word);
        self.taken += 1;
        if self.taken == self.hold_after {
            self.held = true;
        }
    }
    fn set_reference(&mut self, value: u32) {
        self.inner.set_reference(value);
    }
    fn note(&mut self, note: Note) {
        self.inner.note(note);
    }
    fn holds(&self) -> bool {
        self.held
    }
}

#[test]
fn a_held_channel_stops_after_the_word_and_reads_the_next_one_fresh() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    for value in 1..=3 {
        pb.immediate(2, 0x100, value).unwrap();
    }
    rig.start(CMD, pb.words());
    let mut sink = HoldingSink::new(1);
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Held)
    );
    assert_eq!(sink.inner.0, [method_word(2, class_i2m(), 0x100, 1, 0)]);
    assert_eq!(rig.channel.state(), ChannelState::Running);
    // While held, a run reads nothing, even with the next word unmapped.
    rig.mem
        .unmap(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap())
        .unwrap();
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Held)
    );
    assert_eq!(sink.inner.0.len(), 1);
    rig.mem
        .map(
            PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
            CPU_BASE,
        )
        .unwrap();
    // The second word is rewritten while held: the channel reads the new one.
    let mut new = PushBuffer::new();
    new.immediate(2, 0x300, 9).unwrap();
    rig.write(word_va(CMD, 1), new.words());
    sink.held = false;
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::EntryDone)
    );
    assert_eq!(
        sink.inner.0,
        [
            method_word(2, class_i2m(), 0x100, 1, 0),
            method_word(2, class_i2m(), 0x300, 9, 0),
            method_word(2, class_i2m(), 0x100, 3, 0),
        ]
    );
}

#[test]
fn a_hold_on_an_entrys_last_word_ends_the_entry_once_released() {
    let mut rig = make_rig();
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    rig.start(CMD, pb.words());
    let mut sink = HoldingSink::new(1);
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Held)
    );
    assert_eq!(rig.channel.state(), ChannelState::Running);
    assert_eq!(
        rig.channel.end_submission(&mut sink),
        Err(StateError::Running)
    );
    sink.held = false;
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::EntryDone)
    );
    // An idle channel is idle, not held.
    sink.held = true;
    assert_eq!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Idle)
    );
}

#[test]
fn a_parked_channel_reports_the_park_before_a_hold() {
    let mut rig = make_rig();
    rig.set_sem(0);
    let mut pb = PushBuffer::new();
    semaphored(&mut pb, 0, SEM, 1, acquire_op(AcquireOp::Equal));
    rig.start(CMD, pb.words());
    let mut sink = HoldingSink::new(usize::MAX);
    assert!(matches!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Parked(_))
    ));
    sink.held = true;
    assert!(matches!(
        rig.channel.run(&rig.mem, &mut sink, u32::MAX),
        Ok(RunOutcome::Parked(_))
    ));
}

#[test]
fn words_read_counts_executed_words_across_calls_and_entries() {
    // A fresh channel has read nothing.
    let mut rig = make_rig();
    assert_eq!(rig.channel.words_read(), 0);
    // Three immediates: a quantum of 2 executes two (the third, read ahead in the same chunk,
    // is not counted), and the next call the third.
    let mut pb = PushBuffer::new();
    pb.immediate(2, 0x100, 1).unwrap();
    pb.immediate(2, 0x100, 2).unwrap();
    pb.immediate(2, 0x100, 3).unwrap();
    rig.start(CMD, pb.words());
    assert_eq!(
        rig.channel.run(&rig.mem, &mut rig.sink, 2),
        Ok(RunOutcome::QuantumUsed)
    );
    assert_eq!(rig.channel.words_read(), 2);
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.channel.words_read(), 3);
    // A second entry adds its own words; an empty one adds none.
    rig.start(CMD2, &pb.words()[..1]);
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.channel.words_read(), 4);
    rig.start(CMD2, &[]);
    assert_eq!(rig.run(), Ok(RunOutcome::EntryDone));
    assert_eq!(rig.channel.words_read(), 4);
}
