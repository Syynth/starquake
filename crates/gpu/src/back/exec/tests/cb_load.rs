//! Tests of constant-buffer loads through the executor (slice G25): the words reach guest
//! memory where the front's offset put them, across ring entries, and their ranges reach the
//! caches before the next draw, as inline writes do.

use super::*;
use crate::back::cb_load::ConstantBufferError;
use crate::regs::{CONSTANT_BUFFER_LOADS, RegFile, WORD_BYTES};
use crate::ring::MAX_RUN_WORDS;

/// The buffer the front-driven test selects: past every other address the rig uses, with room
/// for [`BIG`] bytes before the end of test memory.
const CB: u64 = BASE + 0x30000;
/// The largest size `SIZE` (16:0) and the 16-bit offset can use together: 64 KiB.
const BIG: u32 = 0x1_0000;

/// `LOAD_CONSTANT_BUFFER(index)` as a ring method number.
fn load(index: u32) -> u16 {
    method_number(&clc797::LOAD_CONSTANT_BUFFER, &[index])
}

/// The ring text selecting the buffer at `buffer` of `size` bytes with upload offset `offset`.
fn select(buffer: u64, size: u32, offset: u32) -> String {
    let c = THREE_D;
    write(c, &clc797::SET_CONSTANT_BUFFER_SELECTOR_A, &[], size)
        + &address(
            c,
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_B,
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_C,
            &[],
            buffer,
        )
        + &write(c, &clc797::LOAD_CONSTANT_BUFFER_OFFSET, &[], offset)
}

/// The ring text of a `Run` of `words` to `LOAD_CONSTANT_BUFFER(index)`.
fn load_run(index: u32, words: &[u32]) -> String {
    let entry = RingEntry::Run {
        channel: CHANNEL,
        class: ClassId(THREE_D),
        method: load(index),
        words: words.into(),
    };
    format!("{entry}\n")
}

/// A load of `words` at `address`, as the front sends it: the selector, the offset 0 and the run.
fn load_at(address: u64, words: &[u32]) -> String {
    select(address, BIG, 0) + &load_run(2, words)
}

/// The page holding `va`.
fn page(va: u64) -> PageRange {
    PageRange::new(va - va % PAGE_SIZE, PAGE_SIZE).unwrap()
}

#[test]
fn a_run_split_across_ring_entries_lands_in_order_and_the_next_continues_after_it() {
    // The front's own register file makes the entries: a run of `MAX_RUN_WORDS` + 3 words is two
    // `Run` entries, each followed by the advanced offset.
    let start = 0x10;
    let words: Vec<u32> = (0..u32::try_from(MAX_RUN_WORDS + 3).unwrap())
        .map(|k| 0xc000_0000 | k)
        .collect();
    let mut front = RegFile::new(CHANNEL, ClassId(THREE_D));
    let mut entries: Vec<RingEntry<'static>> = Vec::new();
    for (method, value) in [
        (&clc797::SET_CONSTANT_BUFFER_SELECTOR_A, BIG),
        (
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_B,
            u32::try_from(CB >> 32).unwrap(),
        ),
        (
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_C,
            u32::try_from(CB & 0xffff_ffff).unwrap(),
        ),
        (&clc797::LOAD_CONSTANT_BUFFER_OFFSET, start),
    ] {
        front.write(method.method(), value, &mut entries).unwrap();
    }
    let method = CONSTANT_BUFFER_LOADS.element(9).unwrap();
    front.write_run(method, &words, &mut entries).unwrap();
    let runs = entries
        .iter()
        .filter(|entry| matches!(entry, RingEntry::Run { .. }))
        .count();
    assert_eq!(runs, 2);
    // The second load continues where the first left the offset.
    let more = [0xdddd_0001, 0xdddd_0002];
    front.write_run(method, &more, &mut entries).unwrap();

    let mut rig = Rig::new();
    rig.run("bind channel=3 subchannel=0 class=0xc797\n");
    for entry in &entries {
        assert_eq!(rig.exec.apply(entry), Ok(()), "{entry}");
    }
    let first = CB + u64::from(start);
    for (k, &word) in (0u64..).zip(words.iter().chain(&more)) {
        assert_eq!(rig.word(first + 4 * k), word, "word {k}");
    }
    // The word before and the word after are untouched.
    assert_eq!(rig.word(first - 4), 0);
    let all = u64::try_from(words.len() + more.len()).unwrap();
    assert_eq!(rig.word(first + 4 * all), 0);
}

#[test]
fn a_load_reaches_the_caches_at_the_next_draw_not_before() {
    // Like an inline write: held back, then handed to the shader cache (and the buffer and
    // texture caches) before the draw.
    let mut rig = Rig::new();
    rig.run(&draw_state());
    rig.run(&load_at(UNIFORMS + 0x40, &[1, 2]));
    assert_eq!(rig.word(UNIFORMS + 0x44), 2);
    assert_eq!(rig.log().invalidated, vec![]);
    rig.run(&draw());
    assert_eq!(rig.log().invalidated, vec![page(UNIFORMS)]);
}

#[test]
fn a_load_drops_a_cached_buffer_it_overlaps_and_not_its_neighbours() {
    // The vertex buffer a draw uploaded is cached over the page at `VERTICES`. As in
    // `the_inline_qmd_is_dropped_from_the_buffer_cache`, two uploads come before the case: the
    // first draw's, and the first after the watch is armed.
    let cases: [(&str, Option<u64>, usize); 4] = [
        ("no load", None, 2),
        ("over the buffer", Some(VERTICES), 3),
        // Ends exactly where the buffer's page starts.
        ("below", Some(VERTICES - 8), 2),
        // Starts exactly where the buffer's page ends.
        ("above", Some(VERTICES + PAGE_SIZE), 2),
    ];
    for (name, at, uploads) in cases {
        let mut rig = Rig::new();
        rig.run(&(draw_state() + &draw()));
        arm_requests(&rig.watch);
        rig.run(&draw());
        if let Some(at) = at {
            rig.run(&load_at(at, &[0x1234_5678, 0x9abc_def0]));
            assert_eq!(rig.word(at + 4), 0x9abc_def0, "{name}");
        }
        rig.run(&draw());
        let snapshot = rig.submit();
        assert_eq!(buffer_uploads(&snapshot), uploads, "{name}: {snapshot}");
    }
}

#[test]
fn a_load_past_the_size_writes_the_words_that_fit_and_keeps_every_word_in_the_mirror() {
    // Offsets `size - 4` (ends at the size: written) and `size` (refused).
    let size = 0x100;
    let mut rig = Rig::new();
    rig.run("bind channel=3 subchannel=0 class=0xc797\n");
    rig.run(&select(CB, size, size - 4));
    let entry = parse_text(&load_run(0, &[0x4444_4444, 0x5555_5555])).unwrap();
    assert_eq!(
        rig.exec.apply(&entry[0]),
        Err(ExecError::ConstantBuffer(ConstantBufferError::PastEnd {
            offset: size,
            size,
        }))
    );
    assert_eq!(rig.exec.counters().failed, 1);
    assert_eq!(rig.word(CB + u64::from(size) - 4), 0x4444_4444);
    assert_eq!(rig.word(CB + u64::from(size)), 0);
    let mirror = rig.exec.mirror(CHANNEL, ClassId(THREE_D)).unwrap();
    assert_eq!(
        mirror.value(u32::from(load(0))),
        Some(0x5555_5555),
        "the mirror stores the words whatever the load did"
    );
    // A later load of one word ending exactly at the size is written.
    rig.run(&load_run(0, &[0x6666_6666]));
    assert_eq!(rig.word(CB + u64::from(size) - 4), 0x6666_6666);
}

#[test]
fn a_load_past_the_size_hands_the_words_it_wrote_to_the_caches_at_the_next_draw() {
    // A load straddling the size writes its first word over the uniforms' page, and that page
    // reaches the caches at the next draw; a load starting at the size writes nothing, and the
    // next draw invalidates nothing.
    let size = 0x44;
    let cases: [(&str, u32, Vec<PageRange>); 2] = [
        ("straddling", size - 4, vec![page(UNIFORMS)]),
        ("from the size", size, vec![]),
    ];
    for (name, offset, invalidated) in cases {
        let mut rig = Rig::new();
        rig.run(&draw_state());
        let first = UNIFORMS + u64::from(offset);
        let before = [rig.word(first), rig.word(first + 4)];
        rig.run(&select(UNIFORMS, size, offset));
        let entry = parse_text(&load_run(1, &[0x7777_0001, 0x7777_0002])).unwrap();
        assert_eq!(
            rig.exec.apply(&entry[0]),
            Err(ExecError::ConstantBuffer(ConstantBufferError::PastEnd {
                offset: size,
                size,
            })),
            "{name}"
        );
        let written = if offset < size {
            [0x7777_0001, before[1]]
        } else {
            before
        };
        assert_eq!([rig.word(first), rig.word(first + 4)], written, "{name}");
        assert_eq!(rig.log().invalidated, vec![], "{name}");
        rig.run(&draw());
        assert_eq!(rig.log().invalidated, invalidated, "{name}");
    }
}

#[test]
fn only_the_3d_class_loads_constant_buffers() {
    // The compute, DMA and 2D classes' method at the same number is not a constant-buffer load,
    // even with the 3D class selecting a buffer on the same channel.
    for (class, subchannel) in [(clc7c0::CLASS, 1), (clc7b5::CLASS, 4), (cl902d::CLASS, 3)] {
        let mut rig = Rig::new();
        rig.run("bind channel=3 subchannel=0 class=0xc797\n");
        rig.run(&select(CB, BIG, 0));
        rig.run(&format!(
            "bind channel=3 subchannel={subchannel} class={class:#x}\n"
        ));
        let entry = RingEntry::Write {
            channel: CHANNEL,
            class: ClassId(class),
            method: load(0),
            value: 0x7777_7777,
        };
        assert_eq!(rig.exec.apply(&entry), Ok(()), "{class:#x}");
        assert_eq!(rig.word(CB), 0, "{class:#x}");
    }
}

#[test]
fn a_split_run_whose_second_entry_passes_the_size_keeps_every_word_that_fits() {
    // The front's own register file splits a run of `MAX_RUN_WORDS` + 3 words into two `Run`
    // entries. The size ends two words into the second entry, so its third word passes it: the
    // first entry is written whole, and of the second entry the two words that fit are written
    // and the third is refused.
    let first_words = u32::try_from(MAX_RUN_WORDS).unwrap();
    let size = (first_words + 2) * WORD_BYTES;
    let words: Vec<u32> = (0..first_words + 3).map(|k| 0xe000_0000 | k).collect();
    let mut front = RegFile::new(CHANNEL, ClassId(THREE_D));
    let mut entries: Vec<RingEntry<'static>> = Vec::new();
    for (method, value) in [
        (&clc797::SET_CONSTANT_BUFFER_SELECTOR_A, size),
        (
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_B,
            u32::try_from(CB >> 32).unwrap(),
        ),
        (
            &clc797::SET_CONSTANT_BUFFER_SELECTOR_C,
            u32::try_from(CB & 0xffff_ffff).unwrap(),
        ),
        (&clc797::LOAD_CONSTANT_BUFFER_OFFSET, 0),
    ] {
        front.write(method.method(), value, &mut entries).unwrap();
    }
    let method = CONSTANT_BUFFER_LOADS.element(4).unwrap();
    front.write_run(method, &words, &mut entries).unwrap();
    let runs: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| matches!(entry, RingEntry::Run { .. }))
        .map(|(k, _)| k)
        .collect();
    assert_eq!(runs.len(), 2);

    let mut rig = Rig::new();
    rig.run("bind channel=3 subchannel=0 class=0xc797\n");
    for entry in &entries[..runs[1]] {
        assert_eq!(rig.exec.apply(entry), Ok(()), "{entry}");
    }
    assert_eq!(
        rig.exec.apply(&entries[runs[1]]),
        Err(ExecError::ConstantBuffer(ConstantBufferError::PastEnd {
            offset: size,
            size,
        }))
    );
    // Every word ending within the size landed: the first entry's and two of the second's.
    let fit = MAX_RUN_WORDS + 2;
    for (k, &word) in (0u64..).zip(&words[..fit]) {
        assert_eq!(rig.word(CB + 4 * k), word, "word {k}");
    }
    // The third word of the second entry, the first past the size, was not written.
    assert_eq!(rig.word(CB + u64::from(size)), 0, "the refused word");
}

#[test]
fn a_constant_buffer_refusal_reads_as_plain_words() {
    let error = ExecError::ConstantBuffer(ConstantBufferError::PastEnd {
        offset: 0x100,
        size: 0x100,
    });
    assert_eq!(
        error.to_string(),
        "constant-buffer load: a constant-buffer word at byte offset 0x100 ends past the \
         buffer's size of 0x100 bytes"
    );
}
