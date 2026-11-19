//! Tests of the constant-buffer upload, over a mirror and a test GPU memory that records each
//! write and can fault after one.

use std::sync::{Mutex, PoisonError};

use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange};

use super::*;
use crate::back::state::Mirror;
use crate::classes::clc797;
use crate::interface::ChannelId;
use crate::memory::TestGpuMemory;
use crate::regs::LOAD_CONSTANT_BUFFER_COUNT;
use crate::ring::ClassId;

/// Where the guest memory behind the GPU mapping starts.
const CPU_BASE: u64 = 0x8000_0000;
/// Where the GPU mapping starts: its upper byte (0x12) is not zero, so a dropped
/// `ADDRESS_UPPER` reads an unmapped address.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Pages mapped: room for a 64 KiB buffer after one page of lower neighbour.
const PAGES: u64 = 32;
/// The buffer most tests select: one page above the mapping's start, so the bytes below it are
/// mapped neighbours.
const BUFFER: u64 = GPU_BASE + PAGE_SIZE;
/// The size most tests select: not a power of two.
const SIZE: u32 = 0x140;
/// What every byte of guest memory holds before a test writes.
const FILL: u8 = 0xa5;
/// Not channel 0.
const CHANNEL: ChannelId = ChannelId(5);
/// The address the injected fault names.
const FAULT_VA: u64 = 0xdead_0000;

/// A test GPU memory that records each write's address and length, and, from `fail_at`,
/// returns [`FAULT_VA`]'s fault after writing.
struct Memory {
    inner: TestGpuMemory<FlatMemory>,
    writes: Mutex<Vec<(u64, usize)>>,
    fail_at: Option<usize>,
}

impl Memory {
    fn new() -> Self {
        Self::failing(None)
    }

    fn failing(fail_at: Option<usize>) -> Self {
        let len = usize::try_from(PAGES * PAGE_SIZE).unwrap();
        let guest = FlatMemory::new(CPU_BASE, len);
        guest.write(CPU_BASE, &vec![FILL; len]).unwrap();
        let inner = TestGpuMemory::new(guest);
        inner
            .map(
                PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
                CPU_BASE,
            )
            .unwrap();
        Self {
            inner,
            writes: Mutex::new(Vec::new()),
            fail_at,
        }
    }

    fn writes(&self) -> Vec<(u64, usize)> {
        self.writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn word(&self, va: u64) -> u32 {
        let mut bytes = [0; 4];
        self.inner.read(va, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }
}

impl GpuMemory for Memory {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        self.inner.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let call = {
            let mut writes = self.writes.lock().unwrap_or_else(PoisonError::into_inner);
            writes.push((va, data.len()));
            writes.len() - 1
        };
        self.inner.write(va, data)?;
        if self.fail_at.is_some_and(|n| call >= n) {
            return Err(GpuFault::Unmapped { va: FAULT_VA });
        }
        Ok(())
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }
}

/// The word every byte of which is [`FILL`].
const FILLED: u32 = u32::from_le_bytes([FILL; 4]);

/// A 3D mirror selecting the buffer at `address` of `size` bytes, with the upload offset
/// `offset` (a whole register word: the bits above the field stay in it).
fn mirror(address: u64, size: u32, offset: u32) -> Mirror {
    let mut mirror = Mirror::new(CHANNEL, ClassId(clc797::CLASS));
    let upper = u32::try_from(address >> u32::BITS).unwrap();
    let lower = u32::try_from(address & u64::from(u32::MAX)).unwrap();
    for (method, value) in [
        (&clc797::SET_CONSTANT_BUFFER_SELECTOR_A, size),
        (&clc797::SET_CONSTANT_BUFFER_SELECTOR_B, upper),
        (&clc797::SET_CONSTANT_BUFFER_SELECTOR_C, lower),
        (&clc797::LOAD_CONSTANT_BUFFER_OFFSET, offset),
    ] {
        mirror.write(method.method(), value).unwrap();
    }
    mirror
}

/// `LOAD_CONSTANT_BUFFER(index)`.
fn load(index: u32) -> u32 {
    clc797::LOAD_CONSTANT_BUFFER.method_at(&[index]).unwrap()
}

/// Runs one load of `words` to `LOAD_CONSTANT_BUFFER(3)`.
fn run(
    mirror: &Mirror,
    words: &[u32],
    memory: &Memory,
    pending: &mut PendingWrites,
) -> Result<(), ConstantBufferError> {
    ConstantBufferLoad::new().method(mirror, load(3), words, memory, pending)
}

/// The pending ranges of one written piece.
fn only(range: Range<u64>) -> Vec<Range<u64>> {
    vec![range]
}

// ---------------------------------------------------------------------------------------------
// Where the words land.

#[test]
fn a_run_lands_at_the_selected_address_plus_the_offset_in_one_write() {
    // Bits above each selector field and above the offset field are set, and ignored.
    let size = 0xfffe_0000 | SIZE;
    let mut m = mirror(BUFFER, size, 0x5a5a_0024);
    let upper_junk = 0xab00_0000 | u32::try_from(BUFFER >> u32::BITS).unwrap();
    m.write(clc797::SET_CONSTANT_BUFFER_SELECTOR_B.method(), upper_junk)
        .unwrap();
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    let words = [0x1111_1111, 0x2222_2222, 0x3333_3333];
    assert_eq!(run(&m, &words, &memory, &mut pending), Ok(()));
    let start = BUFFER + 0x24;
    for (k, word) in (0u64..).zip(words) {
        assert_eq!(memory.word(start + 4 * k), word, "word {k}");
    }
    // The neighbours on both sides are unchanged.
    assert_eq!(memory.word(start - 4), FILLED);
    assert_eq!(memory.word(start + 12), FILLED);
    // One write of the whole run, and its range pending.
    assert_eq!(memory.writes(), [(start, 12)]);
    assert_eq!(pending.ranges(), only(start..start + 12));
    // The mirror is not advanced: the front sends that.
    assert_eq!(
        m.value(clc797::LOAD_CONSTANT_BUFFER_OFFSET.method()),
        Some(0x5a5a_0024)
    );
}

#[test]
fn every_constant_buffer_method_loads_into_the_selected_buffer() {
    for index in 0..LOAD_CONSTANT_BUFFER_COUNT {
        let offset = 0x10 * index;
        let m = mirror(BUFFER, SIZE, offset);
        let memory = Memory::new();
        let mut pending = PendingWrites::new();
        let word = 0x0bad_0000 | index;
        let loaded =
            ConstantBufferLoad::new().method(&m, load(index), &[word], &memory, &mut pending);
        assert_eq!(loaded, Ok(()), "{index}");
        assert_eq!(memory.word(BUFFER + u64::from(offset)), word, "{index}");
    }
}

#[test]
fn other_methods_and_no_words_write_nothing() {
    let m = mirror(BUFFER, SIZE, 0);
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    let past = clc797::LOAD_CONSTANT_BUFFER
        .method_at(&[LOAD_CONSTANT_BUFFER_COUNT])
        .unwrap();
    for method in [
        past,
        clc797::LOAD_CONSTANT_BUFFER_OFFSET.method(),
        clc797::LOAD_INLINE_DATA.method(),
    ] {
        let loaded = ConstantBufferLoad::new().method(&m, method, &[1], &memory, &mut pending);
        assert_eq!(loaded, Ok(()), "{method:#x}");
    }
    assert_eq!(run(&m, &[], &memory, &mut pending), Ok(()));
    assert_eq!(memory.writes(), []);
    assert!(pending.is_empty());
    assert_eq!(memory.word(BUFFER), FILLED);
}

#[test]
fn constant_buffer_loads_are_the_headers_sixteen_methods() {
    // Written out from `classes/clc797.h` (`NVC797_LOAD_CONSTANT_BUFFER(i)`, byte 0x2390 plus 4i)
    // and Eden-NX2's `ASSERT_REG_POSITION(const_buffer, 0x2380)` with `buffer` after four words
    // (`eden-nx2/src/video_core/engines/maxwell_3d.h`): methods 0x8e4 to 0x8f3.
    for method in 0x8e4..=0x8f3 {
        assert!(is_constant_buffer_load(method), "{method:#x}");
    }
    // `LOAD_CONSTANT_BUFFER_OFFSET` below and the method after the sixteenth.
    for method in [0x8e3, 0x8f4] {
        assert!(!is_constant_buffer_load(method), "{method:#x}");
    }
}

#[test]
fn the_selector_and_offset_are_the_headers_registers_and_fields() {
    // Written out from `classes/clc797.h`: `SET_CONSTANT_BUFFER_SELECTOR_A` (0x2380, SIZE 16:0),
    // `_B` (0x2384, ADDRESS_UPPER 7:0), `_C` (0x2388, ADDRESS_LOWER 31:0) and
    // `LOAD_CONSTANT_BUFFER_OFFSET` (0x238c, V 15:0); Eden-NX2's `ConstantBuffer` holds `size`,
    // `address_high`, `address_low` and `offset` in that order.
    let mut m = Mirror::new(CHANNEL, ClassId(clc797::CLASS));
    for (method, value) in [
        (0x2380 / 4, 0xffff_ffff),
        (0x2384 / 4, 0x1234_5678),
        (0x2388 / 4, 0x9abc_def0),
    ] {
        m.write(method, value).unwrap();
    }
    let selected = m.constant_buffer_selector();
    assert_eq!(selected.size, 0x1_ffff);
    assert_eq!(selected.address, 0x78_9abc_def0);
    assert_eq!(clc797::LOAD_CONSTANT_BUFFER_OFFSET.method(), 0x238c / 4);
    assert_eq!(
        clc797::LOAD_CONSTANT_BUFFER_OFFSET_V.get(0xffff_ffff),
        0xffff
    );
}

// ---------------------------------------------------------------------------------------------
// The size bound.

#[test]
fn the_size_edge_accepts_a_word_ending_at_it_and_refuses_one_past() {
    // The last word that fits, then one byte further: refused before anything is written. The
    // bits above `SIZE` (16:0) are set and ignored, so a size read as the whole register word
    // would accept the refused cases.
    let size_word = 0xfffe_0000 | SIZE;
    assert_eq!(
        clc797::SET_CONSTANT_BUFFER_SELECTOR_A_SIZE.get(size_word),
        SIZE
    );
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    let last = SIZE - 4;
    assert_eq!(
        run(
            &mirror(BUFFER, size_word, last),
            &[0x7777_7777],
            &memory,
            &mut pending
        ),
        Ok(())
    );
    assert_eq!(memory.word(BUFFER + u64::from(last)), 0x7777_7777);
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    for offset in [last + 1, SIZE] {
        assert_eq!(
            run(
                &mirror(BUFFER, size_word, offset),
                &[0x7777_7777],
                &memory,
                &mut pending
            ),
            Err(ConstantBufferError::PastEnd { offset, size: SIZE })
        );
    }
    assert_eq!(memory.writes(), []);
    assert!(pending.is_empty());
}

#[test]
fn a_run_whose_last_word_passes_the_size_writes_none_of_its_words() {
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    let start = SIZE - 8;
    let m = mirror(BUFFER, SIZE, start);
    assert_eq!(
        run(&m, &[1, 2, 3], &memory, &mut pending),
        Err(ConstantBufferError::PastEnd {
            offset: SIZE,
            size: SIZE
        })
    );
    assert_eq!(memory.writes(), []);
    assert!(pending.is_empty());
    assert_eq!(memory.word(BUFFER + u64::from(start)), FILLED);
    // The next load, ending exactly at the size, is written whole.
    assert_eq!(run(&m, &[1, 2], &memory, &mut pending), Ok(()));
    let at = BUFFER + u64::from(start);
    assert_eq!(memory.writes(), [(at, 8)]);
    assert_eq!(pending.ranges(), only(at..at + 8));
}

#[test]
fn a_run_with_two_words_past_the_size_reports_the_first() {
    // Both words of the run pass the size, at offsets `SIZE` and `SIZE + 4`: the first in word
    // order is reported, through the upload as well as through `check`.
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    assert_eq!(
        run(&mirror(BUFFER, SIZE, SIZE), &[1, 2], &memory, &mut pending),
        Err(ConstantBufferError::PastEnd {
            offset: SIZE,
            size: SIZE
        })
    );
    assert_eq!(memory.writes(), []);
    assert!(pending.is_empty());
}

#[test]
fn check_compares_each_words_end_with_the_size() {
    // Ends one below, at and one above the size.
    let check_one = |offset: u32| check(BUFFER, SIZE, [offset].into_iter());
    assert_eq!(check_one(SIZE - 5), Ok(()));
    assert_eq!(check_one(SIZE - 4), Ok(()));
    assert_eq!(
        check_one(SIZE - 3),
        Err(ConstantBufferError::PastEnd {
            offset: SIZE - 3,
            size: SIZE
        })
    );
    // A size of zero takes no word; the largest offset is refused whatever the size.
    assert_eq!(
        check(BUFFER, 0, [0].into_iter()),
        Err(ConstantBufferError::PastEnd { offset: 0, size: 0 })
    );
    assert_eq!(
        check(BUFFER, u32::MAX, [u32::MAX].into_iter()),
        Err(ConstantBufferError::PastEnd {
            offset: u32::MAX,
            size: u32::MAX
        })
    );
}

#[test]
fn check_refuses_a_word_ending_past_the_address_space() {
    let offset = 8;
    // The word ends at u64::MAX exactly: representable, accepted.
    let at_end = u64::MAX - u64::from(offset) - 4;
    assert_eq!(check(at_end, SIZE, [offset].into_iter()), Ok(()));
    // One byte further.
    let past = at_end + 1;
    assert_eq!(
        check(past, SIZE, [offset].into_iter()),
        Err(ConstantBufferError::AddressOverflow {
            address: past,
            offset
        })
    );
}

#[test]
fn check_reports_the_first_failing_word_and_the_size_before_the_address() {
    let address = u64::MAX - 2;
    // One word failing both: the size is checked first.
    assert_eq!(
        check(address, SIZE, [SIZE].into_iter()),
        Err(ConstantBufferError::PastEnd {
            offset: SIZE,
            size: SIZE
        })
    );
    // An overflowing word before a word past the size: the first is reported.
    assert_eq!(
        check(address, SIZE, [0, SIZE].into_iter()),
        Err(ConstantBufferError::AddressOverflow { address, offset: 0 })
    );
    // Two words past the size: the first is reported.
    assert_eq!(
        check(BUFFER, SIZE, [4, SIZE + 4, SIZE].into_iter()),
        Err(ConstantBufferError::PastEnd {
            offset: SIZE + 4,
            size: SIZE
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Pieces, faults and divergences.

#[test]
fn a_run_that_wraps_the_offset_field_is_two_writes() {
    // The front advances the offset within its 16-bit field: 0xfff8, 0xfffc, 0, 4.
    let size = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V.max() + 1;
    let start = size - 8;
    let m = mirror(BUFFER, size, start);
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    let words = [0xa1, 0xa2, 0xa3, 0xa4];
    assert_eq!(run(&m, &words, &memory, &mut pending), Ok(()));
    let high = BUFFER + u64::from(start);
    assert_eq!(memory.writes(), [(high, 8), (BUFFER, 8)]);
    assert_eq!(
        [
            memory.word(high),
            memory.word(high + 4),
            memory.word(BUFFER),
            memory.word(BUFFER + 4)
        ],
        words
    );
    // Not at the whole register's next offset, as Eden-NX2 would write it.
    assert_eq!(memory.word(BUFFER + u64::from(size)), FILLED);
    assert_eq!(pending.ranges(), [BUFFER..BUFFER + 8, high..high + 8]);
}

#[test]
fn a_fault_after_a_write_still_invalidates_it_and_stops_the_pieces_after_it() {
    let size = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V.max() + 1;
    let start = size - 8;
    let high = BUFFER + u64::from(start);
    let words = [0xb1, 0xb2, 0xb3, 0xb4];
    let fault = Err(ConstantBufferError::Fault(GpuFault::Unmapped {
        va: FAULT_VA,
    }));
    // The first piece faults after writing: it is pending, the second is not written.
    let memory = Memory::failing(Some(0));
    let mut pending = PendingWrites::new();
    assert_eq!(
        run(&mirror(BUFFER, size, start), &words, &memory, &mut pending),
        fault
    );
    assert_eq!(memory.writes(), [(high, 8)]);
    assert_eq!(memory.word(high), 0xb1);
    assert_eq!(memory.word(BUFFER), FILLED);
    assert_eq!(pending.ranges(), only(high..high + 8));
    // The second piece faults after writing: both are pending.
    let memory = Memory::failing(Some(1));
    let mut pending = PendingWrites::new();
    assert_eq!(
        run(&mirror(BUFFER, size, start), &words, &memory, &mut pending),
        fault
    );
    assert_eq!(memory.writes(), [(high, 8), (BUFFER, 8)]);
    assert_eq!(memory.word(BUFFER), 0xb3);
    assert_eq!(pending.ranges(), [BUFFER..BUFFER + 8, high..high + 8]);
}

#[test]
fn a_piece_over_an_unmapped_page_is_the_exact_fault() {
    // The buffer's second word is on the first page past the mapping: the piece writes nothing
    // (Eden-NX2 would write the mapped first word), and its range is still pending.
    let end = GPU_BASE + PAGES * PAGE_SIZE;
    let address = end - 4;
    let memory = Memory::new();
    let mut pending = PendingWrites::new();
    assert_eq!(
        run(&mirror(address, SIZE, 0), &[1, 2], &memory, &mut pending),
        Err(ConstantBufferError::Fault(GpuFault::Unmapped { va: end }))
    );
    assert_eq!(memory.word(address), FILLED);
    assert_eq!(pending.ranges(), only(address..address + 8));
}

#[test]
fn a_buffer_at_address_zero_is_written_like_any_other() {
    // Eden-NX2 asserts the address is not zero (a log) and writes anyway; so does this layer,
    // where GPU page 0 is mapped.
    let guest = FlatMemory::new(CPU_BASE, usize::try_from(PAGE_SIZE).unwrap());
    let memory = TestGpuMemory::new(guest);
    memory
        .map(PageRange::new(0, PAGE_SIZE).unwrap(), CPU_BASE)
        .unwrap();
    let mut pending = PendingWrites::new();
    let loaded = ConstantBufferLoad::new().method(
        &mirror(0, SIZE, 0x10),
        load(0),
        &[0x5555_aaaa],
        &memory,
        &mut pending,
    );
    assert_eq!(loaded, Ok(()));
    let mut bytes = [0; 4];
    memory.read(0x10, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 0x5555_aaaa);
    assert_eq!(pending.ranges(), only(0x10..0x14));
}

#[test]
fn errors_read_as_plain_words() {
    let cases = [
        (
            ConstantBufferError::PastEnd {
                offset: 0x13d,
                size: 0x140,
            },
            "a constant-buffer word at byte offset 0x13d ends past the buffer's size of 0x140 bytes",
        ),
        (
            ConstantBufferError::PastEnd { offset: 0, size: 0 },
            "a constant-buffer word at byte offset 0x0 ends past the buffer's size of 0x0 bytes",
        ),
        (
            ConstantBufferError::AddressOverflow {
                address: 0xffff_ffff_ffff_fff8,
                offset: 8,
            },
            "a constant-buffer word at byte offset 0x8 of the buffer at 0xfffffffffffffff8 runs \
             past the end of the address space",
        ),
        (
            ConstantBufferError::State(StateError::OutOfRange {
                method: 0xe00,
                limit: 0xe00,
            }),
            "the upload offset could not be read: method 0xe00 is outside this class's register \
             file, which holds methods below 0xe00",
        ),
        (
            ConstantBufferError::Fault(GpuFault::Unmapped { va: 0x1000 }),
            "writing the constant-buffer words faulted: GPU address 0x1000 is not mapped",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}
