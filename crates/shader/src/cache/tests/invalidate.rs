//! Invalidation: the edges of an entry's range on both sides, neighbours, the tracked ranges, the
//! reader's bounds, and what a rewrite costs.

use super::*;
use crate::capture::{MAX_INSTRUCTIONS, ProgramReader, Unreadable};

/// The bytes of a vertex program with one filler: header, filler, `EXIT`.
fn small() -> Vec<u8> {
    program(GuestStage::Vertex, &[1])
}

/// A rig with the small program at [`A`] looked up once; returns its range's end.
fn rig_with_entry() -> (Rig, u64) {
    let mut rig = Rig::new();
    let bytes = small();
    rig.place(A, GuestStage::Vertex, &bytes);
    pending(rig.lookup(&[vertex(A)]));
    (rig, A + bytes.len() as u64)
}

/// Whether the entry at [`A`] is still there: a lookup that captures again means it was not.
fn survived(rig: &mut Rig) -> bool {
    let before = rig.counters().captures;
    rig.lookup(&[vertex(A)]);
    rig.counters().captures == before
}

#[test]
fn a_range_ending_where_an_entry_starts_leaves_it_and_one_byte_more_drops_it() {
    let (mut rig, _) = rig_with_entry();
    rig.cache.invalidate(A - 0x100..A);
    assert!(survived(&mut rig));
    rig.cache.invalidate(A - 0x100..A + 1);
    assert!(!survived(&mut rig));
}

#[test]
fn a_range_ending_where_an_entry_of_the_first_stage_starts_leaves_it() {
    // The first stage in the map's key order sits exactly on the range's exclusive end.
    let mut rig = Rig::new();
    let request = ProgramRequest {
        stage: GuestStage::VertexCullBeforeFetch,
        address: A,
    };
    rig.place(A, request.stage, &small());
    pending(rig.lookup(&[request]));
    rig.cache.invalidate(A - 0x100..A);
    assert_eq!(rig.cache.addresses.len(), 1);
    rig.cache.invalidate(A - 0x100..A + 1);
    assert_eq!(rig.cache.addresses.len(), 0);
}

#[test]
fn a_range_starting_at_an_entrys_end_leaves_it_and_one_byte_earlier_drops_it() {
    let (mut rig, end) = rig_with_entry();
    rig.cache.invalidate(end..end + 0x100);
    assert!(survived(&mut rig));
    rig.cache.invalidate(end - 1..end + 0x100);
    assert!(!survived(&mut rig));
}

#[test]
fn a_range_inside_an_entry_drops_it_and_an_empty_range_drops_nothing() {
    let (mut rig, end) = rig_with_entry();
    // Empty ranges, inside the entry and reversed.
    rig.cache.invalidate(A + 8..A + 8);
    let reversed = end..A;
    rig.cache.invalidate(reversed);
    assert!(survived(&mut rig));
    rig.cache.invalidate(A + 150..A + 151);
    assert!(!survived(&mut rig));
}

#[test]
fn an_invalidation_drops_only_the_entries_it_touches() {
    let mut rig = Rig::new();
    let bytes = small();
    let len = bytes.len() as u64;
    // A neighbour starting exactly at the first entry's end, and one entry further on.
    rig.place(A, GuestStage::Vertex, &bytes);
    rig.place(
        A + len,
        GuestStage::Vertex,
        &program(GuestStage::Vertex, &[2]),
    );
    rig.place(B, GuestStage::Vertex, &program(GuestStage::Vertex, &[3]));
    pending(rig.lookup(&[vertex(A), vertex(A + len), vertex(B)]));
    assert_eq!(rig.counters().captures, 3);
    rig.cache.invalidate(A..A + len);
    pending(rig.lookup(&[vertex(A), vertex(A + len), vertex(B)]));
    // Only the first was captured again.
    assert_eq!(rig.counters().captures, 4);
    assert_eq!(rig.cache.addresses.len(), 3);
}

#[test]
fn an_invalidation_drops_every_stage_captured_at_the_address() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &small());
    pending(rig.lookup(&[vertex(A)]));
    pending(rig.lookup(&[pixel(A)]));
    assert_eq!(rig.cache.addresses.len(), 2);
    rig.cache.invalidate(A..A + 1);
    assert_eq!(rig.cache.addresses.len(), 0);
}

#[test]
fn an_invalidation_reaches_the_last_byte_of_the_longest_program() {
    // A graphics program of the most instructions: its range is `MAX_EXTENT` bytes long.
    let mut memory = Memory::new(usize::try_from(MAX_EXTENT).expect("small") + 0x100);
    let start = BASE;
    let mut bytes = vec![0; header_bytes(GuestStage::Vertex)];
    for tag in 0..MAX_INSTRUCTIONS - 1 {
        bytes.extend_from_slice(&filler(tag).to_le_bytes());
    }
    bytes.extend_from_slice(&exit().to_le_bytes());
    assert_eq!(bytes.len() as u64, MAX_EXTENT);
    memory.write(start, &bytes);
    let mut rig = Rig::with_memory(Limits::SHIPPING, memory);
    pending(rig.lookup(&[vertex(start)]));
    assert_eq!(
        rig.cache.tracked_ranges().first(),
        Some(&(start..start + MAX_EXTENT))
    );
    assert_eq!(rig.cache.tracked_ranges().len(), 1);
    // A write just past it leaves it; a write of its last byte drops it.
    rig.cache
        .invalidate(start + MAX_EXTENT..start + MAX_EXTENT + 1);
    assert_eq!(rig.cache.addresses.len(), 1);
    rig.cache
        .invalidate(start + MAX_EXTENT - 1..start + MAX_EXTENT);
    assert_eq!(rig.cache.addresses.len(), 0);
}

#[test]
fn a_rewrite_of_the_same_bytes_costs_a_capture_not_a_translation() {
    let (mut rig, end) = rig_with_entry();
    rig.executor.run_all();
    let first = ready(rig.lookup(&[vertex(A)]));
    rig.memory.write(A, &small());
    rig.cache.invalidate(A..end);
    let second = ready(rig.lookup(&[vertex(A)]));
    assert_eq!(first[0].0, second[0].0);
    assert_eq!(rig.counters().captures, 2);
    assert_eq!(rig.counters().programs, 1);
    assert_eq!(rig.counters().translations, 1);
    assert_eq!(rig.fake.program_calls(), 1);
    assert_eq!(rig.fake.shader_calls(), 1);
}

#[test]
fn a_rewrite_of_other_bytes_is_a_new_translation() {
    let (mut rig, end) = rig_with_entry();
    rig.executor.run_all();
    let first = ready(rig.lookup(&[vertex(A)]));
    rig.memory.write(A, &program(GuestStage::Vertex, &[9]));
    // Without the invalidation the old program is still found.
    assert_eq!(ready(rig.lookup(&[vertex(A)]))[0].0, first[0].0);
    rig.cache.invalidate(A..end);
    pending(rig.lookup(&[vertex(A)]));
    assert_eq!(rig.counters().translations, 2);
}

#[test]
fn tracked_ranges_are_sorted_and_merged_and_include_refused_reads() {
    let mut rig = Rig::new();
    let bytes = small();
    let len = bytes.len() as u64;
    // Two touching programs, one overlapping the second, one apart, and an unreadable one.
    rig.place(A, GuestStage::Vertex, &bytes);
    rig.place(A + len, GuestStage::Vertex, &bytes);
    rig.place(B, GuestStage::Vertex, &bytes);
    let unreadable = rig.memory.end() - 16;
    rig.memory.write(unreadable, &filler(0).to_le_bytes());
    let request_unreadable = ProgramRequest {
        stage: GuestStage::Compute,
        address: unreadable,
    };
    rig.lookup(&[vertex(B)]);
    rig.lookup(&[request_unreadable]);
    rig.lookup(&[vertex(A + len)]);
    rig.lookup(&[vertex(A)]);
    // A compute capture of the middle program's instructions (no header) overlaps it.
    rig.lookup(&[ProgramRequest {
        stage: GuestStage::Compute,
        address: A + len + 128,
    }]);
    assert_eq!(
        rig.cache.tracked_ranges(),
        [
            A..A + 2 * len,
            B..B + len,
            // The filler and the first unmapped byte the capture tried.
            unreadable..unreadable + 17,
        ]
    );
}

#[test]
fn a_tracked_range_keeps_the_end_of_an_entry_that_contains_a_shorter_one() {
    let mut rig = Rig::new();
    // A vertex program whose header starts with an `EXIT` word: captured as compute from the
    // same address, it ends after 16 bytes, inside the vertex program's range.
    let mut bytes = small();
    bytes[..16].copy_from_slice(&exit().to_le_bytes());
    rig.place(A, GuestStage::Vertex, &bytes);
    rig.lookup(&[vertex(A)]);
    rig.lookup(&[ProgramRequest {
        stage: GuestStage::Compute,
        address: A,
    }]);
    assert_eq!(rig.cache.addresses.len(), 2);
    let ranges = rig.cache.tracked_ranges();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges.first(), Some(&(A..A + bytes.len() as u64)));
}

#[test]
fn a_refused_read_at_a_huge_offset_reaches_at_most_the_longest_capture() {
    let memory = Watched::default();
    let reader = super::super::address::GuestReader::new(&memory, A);
    let mut out = [0; 16];
    assert_eq!(
        reader.read(u64::MAX - 1, &mut out),
        Err(Unreadable {
            offset: u64::MAX - 1
        })
    );
    assert_eq!(reader.end(), A + MAX_EXTENT);
}

/// Memory that reports an unmapped address outside the read it refuses.
struct Liar(u64);

impl ProgramMemory for Liar {
    fn read(&self, _address: u64, _out: &mut [u8]) -> Result<(), Unmapped> {
        Err(Unmapped { address: self.0 })
    }
}

#[test]
fn an_unmapped_address_outside_the_read_counts_as_its_first_byte() {
    for liar in [0, A - 1, A + 0x30, u64::MAX] {
        let memory = Liar(liar);
        let reader = super::super::address::GuestReader::new(&memory, A);
        let mut out = [0; 16];
        assert_eq!(
            reader.read(0x20, &mut out),
            Err(Unreadable { offset: 0x20 }),
            "{liar:#x}"
        );
        assert_eq!(reader.end(), A + 0x21);
    }
    // Inside the read, the reported byte is kept.
    let memory = Liar(A + 0x2f);
    let reader = super::super::address::GuestReader::new(&memory, A);
    let mut out = [0; 16];
    assert_eq!(
        reader.read(0x20, &mut out),
        Err(Unreadable { offset: 0x2f })
    );
    assert_eq!(reader.end(), A + 0x30);
}

/// Memory that records whether it was asked, and answers zeros.
#[derive(Default)]
struct Watched(std::cell::Cell<bool>);

impl ProgramMemory for Watched {
    fn read(&self, _address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        self.0.set(true);
        out.fill(0);
        Ok(())
    }
}

#[test]
fn a_read_reaching_past_the_last_address_is_refused_before_memory_is_asked() {
    let at = u64::MAX - 15;
    let mut out = [0; 16];
    // Bytes `u64::MAX - 15 ..= u64::MAX` fit exactly: the last byte is `u64::MAX`.
    let memory = Watched::default();
    let fits = super::super::address::GuestReader::new(&memory, at);
    assert_eq!(fits.read(0, &mut out), Ok(()));
    assert!(memory.0.get());
    assert_eq!(fits.end(), u64::MAX);
    // One offset more does not, nor an offset itself past the last address; memory is not
    // asked.
    for offset in [1, 16, u64::MAX] {
        let memory = Watched::default();
        let reader = super::super::address::GuestReader::new(&memory, at);
        assert_eq!(
            reader.read(offset, &mut out),
            Err(Unreadable { offset }),
            "{offset:#x}"
        );
        assert!(!memory.0.get(), "{offset:#x}");
        assert_eq!(
            reader.end(),
            at.saturating_add(offset.saturating_add(1).min(MAX_EXTENT))
        );
    }
}

#[test]
fn a_program_at_the_top_of_the_address_space_is_rejected_not_wrapped() {
    let mut rig = Rig::new();
    let at = u64::MAX - 63;
    let rejection = rejected(rig.lookup(&[vertex(at)]));
    assert_eq!(
        *rejection.reason,
        Rejection::Capture(crate::capture::CaptureError::HeaderUnreadable { offset: 0 })
    );
    // Its range is the one byte it tried: the header's read was refused before memory was asked.
    assert_eq!(rig.cache.tracked_ranges().first(), Some(&(at..at + 1)));
    assert_eq!(rig.cache.tracked_ranges().len(), 1);
}
