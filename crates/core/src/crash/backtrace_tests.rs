//! The backtrace: walks over synthetic stacks in flat memory, one per stop condition, the order
//! of the checks with two faults at once, generated chains, and the layouts it adds to
//! `crash.json` and `crash.txt`.

use starquake_memory::FlatMemory;

use super::*;

/// Where the synthetic stack starts.
const STACK_START: u64 = 0x80f0_0000;
/// Its size in bytes.
const STACK_SIZE: usize = 0x1_0000;

/// The stack's end, one past its last byte.
fn stack_end() -> u64 {
    STACK_START + u64::try_from(STACK_SIZE).unwrap()
}

/// The block the walks are given as the stack, as the core loop names a kernel block.
fn stack() -> Mapping {
    Mapping {
        name: "Stack".into(),
        start: STACK_START,
        end: stack_end(),
    }
}

/// Flat memory that holds exactly the stack.
fn stack_memory() -> FlatMemory {
    FlatMemory::new(STACK_START, STACK_SIZE)
}

/// The size of a frame record, as an address step.
fn record_size() -> u64 {
    u64::try_from(FRAME_RECORD_SIZE).unwrap()
}

/// Writes a frame record at `at`: the caller's frame pointer `caller` at its start and the
/// return address `ret` at [`FRAME_RECORD_RETURN`].
fn put_record(memory: &FlatMemory, at: u64, caller: u64, ret: u64) {
    memory.write(at, &caller.to_le_bytes()).unwrap();
    let ret_at = at + u64::try_from(FRAME_RECORD_RETURN).unwrap();
    memory.write(ret_at, &ret.to_le_bytes()).unwrap();
}

/// One module, `main`, at `0x8000_0000` for `0x1000` bytes.
fn modules() -> Vec<Module> {
    vec![Module {
        name: "main".into(),
        base: 0x8000_0000,
        size: 0x1000,
        build_id: vec![0xab, 0x01, 0xff],
    }]
}

/// Registers with the program counter `pc`, the link register `lr` and the frame pointer `fp`;
/// every other register holds a value no walk should use.
fn regs(pc: u64, lr: u64, fp: u64) -> GuestRegs {
    let mut regs = GuestRegs {
        pc,
        sp: 0xdead_0000,
        ..GuestRegs::default()
    };
    for (index, x) in (0u64..).zip(regs.x.iter_mut()) {
        *x = 0xbad0_0000 + index;
    }
    regs.x[LINK_REGISTER] = lr;
    regs.x[FRAME_POINTER_REGISTER] = fp;
    regs
}

/// A frame at `address`, inside `main` when `offset` says so, read from `record` if any.
fn frame(address: u64, offset: Option<u64>, record: Option<u64>) -> Frame {
    Frame {
        address,
        location: offset.map(|offset| Location {
            module: "main".into(),
            offset,
        }),
        record,
    }
}

/// The two frames every walk starts with, from [`regs`]`(0x8000_00b8, 0x8000_0124, _)`.
fn head() -> [Frame; 2] {
    [
        frame(0x8000_00b8, Some(0xb8), None),
        frame(0x8000_0124, Some(0x124), None),
    ]
}

/// Walks from `fp` with the pc and lr of [`head`].
fn walk(fp: u64, stack: Option<Mapping>, memory: &FlatMemory) -> Backtrace {
    Backtrace::walk(
        &regs(0x8000_00b8, 0x8000_0124, fp),
        stack,
        memory,
        &modules(),
    )
}

/// The chain the full-chain tests walk, its records written by hand byte by byte (not by
/// [`put_record`]): at `0x80f0_ff00`, the caller's frame pointer `0x80f0_ff40` and the return
/// address `0x8000_0200`; at `0x80f0_ff40`, a null frame pointer and the return address
/// `0x7000_0040`, which is in no module.
fn hand_built_chain() -> FlatMemory {
    let memory = stack_memory();
    memory
        .write(
            0x80f0_ff00,
            &[
                0x40, 0xff, 0xf0, 0x80, 0, 0, 0, 0, // the caller's frame pointer
                0x00, 0x02, 0x00, 0x80, 0, 0, 0, 0, // the return address
            ],
        )
        .unwrap();
    memory
        .write(
            0x80f0_ff40,
            &[
                0, 0, 0, 0, 0, 0, 0, 0, // a null frame pointer
                0x40, 0x00, 0x00, 0x70, 0, 0, 0, 0, // the return address
            ],
        )
        .unwrap();
    memory
}

#[test]
fn a_full_chain_is_walked_to_its_null_frame_pointer() {
    let backtrace = walk(0x80f0_ff00, Some(stack()), &hand_built_chain());
    let [pc, lr] = head();
    assert_eq!(
        backtrace,
        Backtrace {
            stack: Some(stack()),
            frames: vec![
                pc,
                lr,
                frame(0x8000_0200, Some(0x200), Some(0x80f0_ff00)),
                frame(0x7000_0040, None, Some(0x80f0_ff40)),
            ],
            end: BacktraceEnd::NullFramePointer,
        }
    );
}

#[test]
fn a_null_frame_pointer_in_the_registers_leaves_the_pc_and_the_lr() {
    let backtrace = walk(0, Some(stack()), &stack_memory());
    assert_eq!(backtrace.frames, head());
    assert_eq!(backtrace.end, BacktraceEnd::NullFramePointer);
}

#[test]
fn a_misaligned_frame_pointer_ends_the_walk_and_one_on_the_alignment_is_followed() {
    let memory = stack_memory();
    // A multiple of the alignment that is not a multiple of eight is followed; a frame pointer
    // two bytes past one is not.
    let aligned = 0x80f0_ff00 + FRAME_POINTER_ALIGN;
    let misaligned = 0x80f0_ff40 + FRAME_POINTER_ALIGN / 2;
    put_record(&memory, aligned, misaligned, 0x8000_0300);
    let backtrace = walk(aligned, Some(stack()), &memory);
    let [pc, lr] = head();
    assert_eq!(
        backtrace.frames,
        [pc, lr, frame(0x8000_0300, Some(0x300), Some(aligned))]
    );
    assert_eq!(
        backtrace.end,
        BacktraceEnd::Misaligned {
            frame_pointer: misaligned
        }
    );
    // Every residue but zero is refused.
    for offset in 1..FRAME_POINTER_ALIGN {
        let fp = 0x80f0_ff00 + offset;
        assert_eq!(
            walk(fp, Some(stack()), &memory).end,
            BacktraceEnd::Misaligned { frame_pointer: fp }
        );
    }
}

#[test]
fn a_record_outside_the_stack_ends_the_walk() {
    let memory = stack_memory();
    let size = record_size();
    // The last record that fits ends exactly at the stack's end, and is read; the next, which
    // would run eight bytes past the end, is not.
    let last = stack_end() - size;
    let straddling = stack_end() - size / 2;
    put_record(&memory, last, straddling, 0x8000_0400);
    let backtrace = walk(last, Some(stack()), &memory);
    assert_eq!(backtrace.frames.len(), head().len() + 1);
    assert_eq!(backtrace.frames[2].record, Some(last));
    assert_eq!(
        backtrace.end,
        BacktraceEnd::OutsideStack {
            frame_pointer: straddling
        }
    );
    // The first record of the stack is read; one just below it is not.
    let below = STACK_START - size;
    put_record(&memory, STACK_START, below, 0x8000_0404);
    let backtrace = walk(STACK_START, Some(stack()), &memory);
    assert_eq!(backtrace.frames[2].record, Some(STACK_START));
    assert_eq!(
        backtrace.end,
        BacktraceEnd::OutsideStack {
            frame_pointer: below
        }
    );
    // With no stack, the first record is outside it.
    let none = walk(0x80f0_ff00, None, &memory);
    assert_eq!(none.frames, head());
    assert_eq!(
        none.end,
        BacktraceEnd::OutsideStack {
            frame_pointer: 0x80f0_ff00
        }
    );
    assert_eq!(none.stack, None);
}

#[test]
fn a_record_that_would_end_past_the_top_of_the_address_space_is_outside_the_stack() {
    // A stack whose end is the last aligned address of the 64-bit space. The record that ends
    // exactly at its end is inside it (and the flat memory, which holds nothing there, refuses
    // its read); the next aligned one would end past 2^64, and is outside it, without a panic.
    let end = u64::MAX - (FRAME_POINTER_ALIGN - 1);
    let top = Mapping {
        name: "Stack".into(),
        start: end - 0x1000,
        end,
    };
    let memory = stack_memory();
    let size = record_size();
    let fits = end - size;
    assert_eq!(
        walk(fits, Some(top.clone()), &memory).end,
        BacktraceEnd::Unreadable {
            frame_pointer: fits
        }
    );
    let wrapping = fits + FRAME_POINTER_ALIGN;
    assert!(wrapping.checked_add(size).is_none());
    assert_eq!(
        walk(wrapping, Some(top), &memory).end,
        BacktraceEnd::OutsideStack {
            frame_pointer: wrapping
        }
    );
}

#[test]
fn a_chain_that_loops_ends_at_the_first_record_read_twice() {
    let memory = stack_memory();
    let (a, b) = (0x80f0_ff00, 0x80f0_ff40);
    put_record(&memory, a, b, 0x8000_0500);
    put_record(&memory, b, a, 0x8000_0504);
    let backtrace = walk(a, Some(stack()), &memory);
    let [pc, lr] = head();
    assert_eq!(
        backtrace.frames,
        [
            pc,
            lr,
            frame(0x8000_0500, Some(0x500), Some(a)),
            frame(0x8000_0504, Some(0x504), Some(b)),
        ]
    );
    assert_eq!(backtrace.end, BacktraceEnd::Loop { frame_pointer: a });
    // A record that names itself.
    put_record(&memory, a, a, 0x8000_0508);
    let itself = walk(a, Some(stack()), &memory);
    assert_eq!(itself.frames.len(), head().len() + 1);
    assert_eq!(itself.end, BacktraceEnd::Loop { frame_pointer: a });
}

/// Writes a chain of `count` records, one after another from the stack's start, the last
/// naming `last_caller`; record `i` returns to `0x8000_0000 + 4 * i`. Returns the first's
/// address.
fn straight_chain(memory: &FlatMemory, count: usize, last_caller: u64) -> u64 {
    let size = record_size();
    for index in 0..count {
        let index = u64::try_from(index).unwrap();
        let at = STACK_START + index * size;
        let caller = if index + 1 == u64::try_from(count).unwrap() {
            last_caller
        } else {
            at + size
        };
        put_record(memory, at, caller, 0x8000_0000 + 4 * index);
    }
    STACK_START
}

#[test]
fn the_walk_reads_at_most_its_limit_of_records() {
    // Exactly the limit, ending in a null frame pointer: the chain is complete.
    let memory = stack_memory();
    let first = straight_chain(&memory, MAX_FRAME_RECORDS, 0);
    let complete = walk(first, Some(stack()), &memory);
    assert_eq!(complete.end, BacktraceEnd::NullFramePointer);
    assert_eq!(complete.frames.len(), head().len() + MAX_FRAME_RECORDS);
    // One more record: the walk stops after the limit, with the limit's frames.
    let memory = stack_memory();
    let size = record_size();
    let past = STACK_START + u64::try_from(MAX_FRAME_RECORDS).unwrap() * size;
    let first = straight_chain(&memory, MAX_FRAME_RECORDS + 1, 0);
    let limited = walk(first, Some(stack()), &memory);
    assert_eq!(limited.end, BacktraceEnd::DepthLimit);
    assert_eq!(limited.frames.len(), head().len() + MAX_FRAME_RECORDS);
    assert_eq!(limited.frames.last().unwrap().record, Some(past - size));
    // Every record read is a frame, in chain order.
    for (index, frame) in (0u64..).zip(&limited.frames[head().len()..]) {
        assert_eq!(frame.record, Some(STACK_START + index * size));
        assert_eq!(frame.address, 0x8000_0000 + 4 * index);
    }
}

#[test]
fn a_record_the_memory_refuses_ends_the_walk_with_a_note() {
    // The stack block is twice what the flat memory holds: a record in its upper half is inside
    // the stack but cannot be read.
    let memory = stack_memory();
    let unreadable = stack_end();
    put_record(&memory, 0x80f0_ff00, unreadable, 0x8000_0600);
    let backtrace = walk(0x80f0_ff00, Some(block_of_twice()), &memory);
    assert_eq!(backtrace.frames.len(), head().len() + 1);
    assert_eq!(
        backtrace.end,
        BacktraceEnd::Unreadable {
            frame_pointer: unreadable
        }
    );
    // A record whose last byte is past the memory's end is refused whole.
    let straddling = stack_end() - record_size() / 2;
    let backtrace = walk(straddling, Some(block_of_twice()), &memory);
    assert_eq!(
        backtrace.end,
        BacktraceEnd::Unreadable {
            frame_pointer: straddling
        }
    );
}

/// [`stack`], twice as long as the flat memory that backs it.
fn block_of_twice() -> Mapping {
    Mapping {
        end: stack_end() + u64::try_from(STACK_SIZE).unwrap(),
        ..stack()
    }
}

#[test]
fn the_checks_come_in_their_documented_order_when_two_faults_are_present() {
    let memory = stack_memory();
    // Null, with no stack: null.
    assert_eq!(walk(0, None, &memory).end, BacktraceEnd::NullFramePointer);
    // Misaligned and outside the stack: misaligned.
    let outside_and_misaligned = 0x10 + FRAME_POINTER_ALIGN / 2;
    assert_eq!(
        walk(outside_and_misaligned, Some(stack()), &memory).end,
        BacktraceEnd::Misaligned {
            frame_pointer: outside_and_misaligned
        }
    );
    // Outside the stack and unreadable: outside.
    let outside_and_unreadable = stack_end() + 0x100;
    assert_eq!(
        walk(outside_and_unreadable, Some(stack()), &memory).end,
        BacktraceEnd::OutsideStack {
            frame_pointer: outside_and_unreadable
        }
    );
    // A loop at the limit: the loop. The chain's last record names its first, after exactly the
    // limit of records.
    let looped = stack_memory();
    let first = straight_chain(&looped, MAX_FRAME_RECORDS, STACK_START);
    assert_eq!(
        walk(first, Some(stack()), &looped).end,
        BacktraceEnd::Loop {
            frame_pointer: first
        }
    );
    // The limit and an unreadable record: the limit. The last record names one in the half of
    // the block the memory does not hold.
    let limited = stack_memory();
    let first = straight_chain(&limited, MAX_FRAME_RECORDS, stack_end());
    assert_eq!(
        walk(first, Some(block_of_twice()), &limited).end,
        BacktraceEnd::DepthLimit
    );
}

/// A fixed-seed generator (`SplitMix64`), so the generated cases are the same on every run.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `bound`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[test]
fn generated_chains_walk_back_to_what_was_written() {
    // Records at distinct random slots of the stack, chained in a random order, each with a
    // random return address that is never a frame pointer the chain uses: reading a record's
    // two words the other way round, or skipping one, gives other frames.
    let mut numbers = Numbers(0x5eed_0206);
    let slots = u64::try_from(STACK_SIZE).unwrap() / record_size();
    for case in 0..200 {
        let memory = stack_memory();
        let count = usize::try_from(numbers.below(40)).unwrap();
        let mut used = BTreeSet::new();
        let mut chain = Vec::new();
        while chain.len() < count {
            let at = STACK_START + numbers.below(slots) * record_size();
            if used.insert(at) {
                chain.push(at);
            }
        }
        let returns: Vec<u64> = (0..count)
            .map(|_| 0x8000_0000 + numbers.below(0x2000) * 4 + 1)
            .collect();
        for (index, &at) in chain.iter().enumerate() {
            let caller = chain.get(index + 1).copied().unwrap_or(0);
            put_record(&memory, at, caller, returns[index]);
        }
        let first = chain.first().copied().unwrap_or(0);
        let backtrace = walk(first, Some(stack()), &memory);
        let mut expected = head().to_vec();
        for (&at, &ret) in chain.iter().zip(&returns) {
            expected.push(Frame {
                address: ret,
                location: Location::of(ret, &modules()),
                record: Some(at),
            });
        }
        assert_eq!(backtrace.frames, expected, "case {case}");
        assert_eq!(backtrace.end, BacktraceEnd::NullFramePointer, "case {case}");
    }
}

#[test]
fn a_stack_of_noise_ends_the_walk_within_its_limit_without_a_panic() {
    // Random bytes in the stack and random frame pointers, many of them inside it: whatever
    // the chain, the walk ends with at most the limit's frames, each read from inside the stack.
    let mut numbers = Numbers(0x0bad_57ac);
    for case in 0..300 {
        let memory = stack_memory();
        let mut bytes = vec![0u8; STACK_SIZE];
        for chunk in bytes.chunks_mut(8) {
            // Half of the words point back into the stack, aligned or not.
            let word = if numbers.below(2) == 0 {
                STACK_START + numbers.below(u64::try_from(STACK_SIZE).unwrap())
            } else {
                numbers.next()
            };
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        memory.write(STACK_START, &bytes).unwrap();
        let fp = STACK_START + numbers.below(u64::try_from(STACK_SIZE).unwrap());
        let backtrace = walk(fp, Some(stack()), &memory);
        assert!(
            backtrace.frames.len() <= head().len() + MAX_FRAME_RECORDS,
            "case {case}"
        );
        for frame in &backtrace.frames[head().len()..] {
            let record = frame.record.expect("a frame from a record");
            assert!(
                record >= STACK_START && record + record_size() <= stack_end(),
                "case {case}: {record:#x}"
            );
        }
    }
}

#[test]
fn the_backtrace_json_layout_is_pinned() {
    // Every field distinct and non-zero: a stack, the two register frames and one read from a
    // record (one placed, one in no module), and an end that names a frame pointer.
    let memory = stack_memory();
    put_record(&memory, 0x80f0_ff00, 0x80f0_ff42, 0x7000_0010);
    let backtrace = walk(0x80f0_ff00, Some(stack()), &memory);
    assert_eq!(
        serde_json::to_string_pretty(&backtrace).unwrap(),
        r#"{
  "stack": {
    "name": "Stack",
    "start": "0x80f00000",
    "end": "0x80f10000"
  },
  "frames": [
    {
      "address": "0x800000b8",
      "location": {
        "module": "main",
        "offset": "0xb8"
      },
      "record": null
    },
    {
      "address": "0x80000124",
      "location": {
        "module": "main",
        "offset": "0x124"
      },
      "record": null
    },
    {
      "address": "0x70000010",
      "location": null,
      "record": "0x80f0ff00"
    }
  ],
  "end": {
    "reason": "misaligned",
    "frame_pointer": "0x80f0ff42"
  }
}"#
    );
    // With no stack, the key is written as `null`.
    let none = serde_json::to_value(walk(0, None, &memory)).unwrap();
    assert_eq!(
        none.as_object().unwrap().get("stack"),
        Some(&serde_json::Value::Null)
    );
}

#[test]
fn every_way_a_walk_ends_is_said_in_words_and_pinned_as_json() {
    let cases = [
        (
            BacktraceEnd::NullFramePointer,
            "the chain ends at a null frame pointer",
            serde_json::json!({ "reason": "null_frame_pointer" }),
        ),
        (
            BacktraceEnd::Misaligned {
                frame_pointer: 0x80f0_ff02,
            },
            "cut short: the frame pointer 0x80f0ff02 is not a multiple of 4",
            serde_json::json!({ "reason": "misaligned", "frame_pointer": "0x80f0ff02" }),
        ),
        (
            BacktraceEnd::Loop {
                frame_pointer: 0x80f0_ff10,
            },
            "cut short: the frame record at 0x80f0ff10 was read before (the chain loops)",
            serde_json::json!({ "reason": "loop", "frame_pointer": "0x80f0ff10" }),
        ),
        (
            BacktraceEnd::OutsideStack {
                frame_pointer: 0x7f00_0020,
            },
            "cut short: the frame record at 0x7f000020 is outside the stack",
            serde_json::json!({ "reason": "outside_stack", "frame_pointer": "0x7f000020" }),
        ),
        (
            BacktraceEnd::DepthLimit,
            "cut short: 256 frame records were read, the most a backtrace reads",
            serde_json::json!({ "reason": "depth_limit" }),
        ),
        (
            BacktraceEnd::Unreadable {
                frame_pointer: 0x80f1_0030,
            },
            "cut short: the frame record at 0x80f10030 could not be read",
            serde_json::json!({ "reason": "unreadable", "frame_pointer": "0x80f10030" }),
        ),
    ];
    for (end, text, json) in cases {
        assert_eq!(end.text(), text);
        assert_eq!(serde_json::to_value(end).unwrap(), json);
    }
    // The texts above name the shipping limits as they are.
    assert_eq!((MAX_FRAME_RECORDS, FRAME_POINTER_ALIGN), (256, 4));
}

/// A crash report with the hand-built chain's backtrace, as the text tests read it: four
/// registers kept, so the golden text stays short.
fn report_with(stack: Option<Mapping>, fp: u64, memory: &FlatMemory) -> CrashReport {
    let regs = regs(0x8000_00b8, 0x8000_0124, fp);
    let backtrace = Backtrace::walk(&regs, stack, memory, &modules());
    let mut report = CrashReport::new(
        Exception::UndefinedInstruction,
        &regs,
        &[Mapping {
            name: "stack".into(),
            start: STACK_START,
            end: stack_end(),
        }],
        &modules(),
        vec!["#0 core 0 thread 1: lifecycle ProcessStart 1".into()],
        &["starquake".into(), "run".into(), "crash.nro".into()],
    )
    .with_backtrace(backtrace);
    report.emulator = "starquake 9.9.9".into();
    report.registers.x.truncate(4);
    report
}

#[test]
fn the_crash_text_with_a_full_chain_is_pinned() {
    let report = report_with(Some(stack()), 0x80f0_ff00, &hand_built_chain());
    assert_eq!(
        report.to_text(),
        "\
starquake 9.9.9 crash report
command line: starquake run crash.nro

exception: undefined instruction
instruction: pc 0x800000b8 (main+0xb8)

backtrace, innermost first:
  #0 0x800000b8 (main+0xb8), pc
  #1 0x80000124 (main+0x124), lr
  #2 0x80000200 (main+0x200), frame record at 0x80f0ff00
  #3 0x70000040 (in no module), frame record at 0x80f0ff40
  stack: 0x80f00000-0x80f10000 Stack
  the chain ends at a null frame pointer

registers:
  x0  0x00000000bad00000 x1  0x00000000bad00001 x2  0x00000000bad00002 x3  0x00000000bad00003
  sp  0x00000000dead0000 pc  0x00000000800000b8 pstate 0x0
  tpidr 0x0000000000000000 tpidrro 0x0000000000000000

memory map:
  0x80f00000-0x80f10000 stack

modules:
  main base 0x80000000 size 0x1000 build id ab01ff

last events, oldest first:
  #0 core 0 thread 1: lifecycle ProcessStart 1
"
    );
}

#[test]
fn the_crash_text_says_how_a_chain_was_cut_short_and_when_there_is_no_stack() {
    let memory = stack_memory();
    put_record(&memory, 0x80f0_ff00, 0x80f0_ff42, 0x8000_0200);
    let cut = report_with(Some(stack()), 0x80f0_ff00, &memory).to_text();
    assert!(
        cut.contains(
            "  #2 0x80000200 (main+0x200), frame record at 0x80f0ff00\n\
             \x20 stack: 0x80f00000-0x80f10000 Stack\n\
             \x20 cut short: the frame pointer 0x80f0ff42 is not a multiple of 4\n\nregisters:\n"
        ),
        "{cut}"
    );
    let none = report_with(None, 0x80f0_ff00, &memory).to_text();
    assert!(
        none.contains(
            "  #1 0x80000124 (main+0x124), lr\n\
             \x20 stack: none (the stack pointer is in no memory the guest can read and write)\n\
             \x20 cut short: the frame record at 0x80f0ff00 is outside the stack\n"
        ),
        "{none}"
    );
    // A report built without a backtrace has no such section.
    let mut without = report_with(None, 0, &memory);
    without.backtrace = None;
    assert!(!without.to_text().contains("backtrace"));
}
