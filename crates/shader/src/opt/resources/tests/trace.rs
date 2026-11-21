//! Tracing: each traced form, its offsets, the edges of its bounds, and the handle a
//! descriptor's words give against the handle the interpreter computes.

use std::fmt::Write as _;

use super::{BOUND_BANK, buffer, parse, sample_of, sources, untraced, value};
use crate::interp::Machine;
use crate::ir::Operand;
use crate::opt::resources::trace::dynamic_size_shift;
use crate::opt::resources::{
    DESCRIPTOR_SIZE_SHIFT, MAX_TRACE_DEPTH, MIN_DYNAMIC_SIZE_SHIFT, Untraced, scan,
};
use crate::translate::float::CBUF_BANKS;

#[test]
fn offsets_are_aligned_as_the_read_is() {
    let text = sample_of("    %0: u32 = GetCbufU32 u32 3, u32 0x13\n", "%0");
    assert_eq!(sources(&text), "%60: cbuf 3 at 0x10\n");
    // A pair element is its word of the eight bytes the read covers.
    for (offset, element, word) in [(0x1c, 1, 0x1c), (0x1c, 0, 0x18), (0x20, 1, 0x24)] {
        let lines = format!(
            "    %0: u32x2 = GetCbufU32x2 u32 3, u32 {offset:#x}\n    \
             %1: u32 = CompositeExtractU32x2 %0, u32 {element:#x}\n"
        );
        let text = sample_of(&lines, "%1");
        assert_eq!(sources(&text), format!("%60: cbuf 3 at {word:#x}\n"));
    }
}

#[test]
fn shift_widths_at_the_edge() {
    let at = |shift: u32| {
        sample_of(
            &format!(
                "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    \
                 %1: u32 = ShiftLeftLogical32 %0, u32 {shift:#x}\n"
            ),
            "%1",
        )
    };
    let last = u32::BITS - 1;
    assert_eq!(
        sources(&at(last)),
        format!("%60: cbuf 3 at 0x10 << {last} & 0x80000000\n")
    );
    assert_eq!(
        untraced(&at(u32::BITS)),
        Untraced::ShiftTooWide {
            at: value(1),
            shift: u32::BITS
        }
    );
}

#[test]
fn banks_at_the_edge() {
    let last = u32::from(CBUF_BANKS) - 1;
    let read = |bank: u32| {
        sample_of(
            &format!("    %0: u32 = GetCbufU32 u32 {bank:#x}, u32 0x10\n"),
            "%0",
        )
    };
    assert_eq!(sources(&read(last)), format!("%60: cbuf {last} at 0x10\n"));
    assert_eq!(untraced(&read(last + 1)), Untraced::BankPastLast(last + 1));
    // An index word in the last bank is kept; one past it is not, and the array is indexed at
    // run time instead.
    let indexed = |bank: u32| {
        sample_of(
            &format!(
                "    %0: u32 = GetCbufU32 u32 {bank:#x}, u32 0x8\n    \
                 %1: u32 = ShiftLeftLogical32 %0, u32 0x3\n    \
                 %2: u32 = IAdd32 %1, u32 0x100\n    \
                 %3: u32 = GetCbufU32 u32 0x3, %2\n"
            ),
            "%3",
        )
    };
    assert_eq!(
        sources(&indexed(last)),
        format!(
            "%60: cbuf 3 at 0x100, 1024 handles 8 bytes apart, index from cbuf {last} at 0x8 << 3\n"
        )
    );
    assert_eq!(
        sources(&indexed(last + 1)),
        "%60: cbuf 3 at 0x100, 1024 handles 8 bytes apart\n"
    );
}

/// A handle masked `depth` times: its read is `depth` operations below the handle.
fn masked(depth: u32) -> String {
    let mut lines = String::from("    %0: u32 = GetCbufU32 u32 3, u32 0x10\n");
    for k in 1..=depth {
        writeln!(
            lines,
            "    %{k}: u32 = BitwiseAnd32 %{}, u32 0xffffffff",
            k - 1
        )
        .expect("writes");
    }
    sample_of(&lines, &format!("%{depth}"))
}

#[test]
fn the_depth_limit_at_its_edge() {
    assert_eq!(sources(&masked(MAX_TRACE_DEPTH)), "%60: cbuf 3 at 0x10\n");
    assert_eq!(untraced(&masked(MAX_TRACE_DEPTH + 1)), Untraced::TooDeep);
    // The work of one fixed input: the three listed instructions and the trace's two steps.
    let text = sample_of(
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = BitwiseAnd32 %0, u32 0xff\n",
        "%1",
    );
    let found = scan(&parse(&text), buffer(BOUND_BANK)).expect("scans");
    // One block in the block order, four listed instructions, two trace steps.
    assert_eq!(found.work().steps(), 1 + 4 + 2);
}

/// Each traced handle form, as lines computing `%h`: the same lines with the handle stored in
/// `r0` run in the interpreter, and the descriptor's words must give the same handle.
const FORMS: [(&str, &str); 9] = [
    ("    %0: u32 = GetCbufU32 u32 3, u32 0x10\n", "%0"),
    (
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = ShiftLeftLogical32 %0, u32 0x14\n",
        "%1",
    ),
    (
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = BitwiseAnd32 %0, u32 0xfff00\n",
        "%1",
    ),
    // A mask, then a shift: the mask moves with the value.
    (
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = BitwiseAnd32 u32 0xfff, %0\n    \
         %2: u32 = ShiftLeftLogical32 %1, u32 0x14\n",
        "%2",
    ),
    // A shift, then a mask: the mask applies after the shift.
    (
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = ShiftLeftLogical32 %0, u32 0x4\n    \
         %2: u32 = BitwiseAnd32 %1, u32 0xff0f0\n",
        "%2",
    ),
    (
        "    %0: u32 = GetCbufU32 u32 3, u32 0x10\n    %1: u32 = BitwiseAnd32 %0, u32 0xfffff\n    \
         %2: u32 = GetCbufU32 u32 4, u32 0x8\n    %3: u32 = ShiftLeftLogical32 %2, u32 0x14\n    \
         %4: u32 = BitwiseOr32 %3, %1\n",
        "%4",
    ),
    (
        "    %0: u32x2 = GetCbufU32x2 u32 3, u32 0x18\n    %1: u32 = CompositeExtractU32x2 %0, u32 0x1\n",
        "%1",
    ),
    (
        "    %0: u32x2 = GetCbufU32x2 u32 3, u32 0x18\n    %1: u32 = CompositeExtractU32x2 %0, u32 0x0\n",
        "%1",
    ),
    (
        "    %0: u32 = GetCbufU32 u32 4, u32 0xc\n    %1: opaque = Identity %0\n",
        "%1",
    ),
];

/// Constant buffers 3 and 4 with distinct words: word `k` of bank `b` is a mix of both.
fn machine() -> Machine {
    let mut machine = Machine::new();
    for bank in [3_u32, 4] {
        let bytes = (0..16_u32)
            .flat_map(|k| (0x9e37_79b9_u32.wrapping_mul(k + 1) ^ (bank << 28)).to_le_bytes())
            .collect();
        machine.cbufs.insert(bank, bytes);
    }
    machine
}

/// The word at `offset` of `bank` in `machine`.
fn word_of(machine: &Machine, bank: u8, offset: u32) -> Option<u32> {
    let bytes = machine.cbufs.get(&u32::from(bank))?;
    let start = usize::try_from(offset).ok()?;
    let word: [u8; 4] = bytes.get(start..start + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(word))
}

#[test]
fn descriptor_handles_match_the_interpreter() {
    let start = machine();
    for (lines, handle) in FORMS {
        let found = scan(&parse(&sample_of(lines, handle)), buffer(BOUND_BANK)).expect("scans");
        let (_, source) = found.sources().next().expect("one texture");
        let traced = source
            .handle(|bank, offset| word_of(&start, bank, offset))
            .expect("words present");
        let program = parse(&format!(
            "block0:\n{lines}    %40 = SetRegister r0, {handle}\n    return\n"
        ));
        let mut run = start.clone();
        run.run(&program).expect("runs");
        assert_eq!(traced, run.regs[0], "{lines}");
    }
}

/// Pinned for the conflict with PR #4588 (module docs): a mask is kept and applied after the
/// shift, so the handle is the IR's, not `word << trailing_zeros(mask)`.
#[test]
fn a_mask_is_applied_after_the_shift() {
    let (lines, handle) = FORMS[2];
    assert_eq!(
        sources(&sample_of(lines, handle)),
        "%60: cbuf 3 at 0x10 & 0xfff00\n"
    );
    let (lines, handle) = FORMS[3];
    assert_eq!(
        sources(&sample_of(lines, handle)),
        "%60: cbuf 3 at 0x10 << 20 & 0xfff00000\n"
    );
}

#[test]
fn the_smallest_spacing_is_four_bytes() {
    // The spacing an index scaled by `shift` gives (Eden's `DynamicDescriptorSizeShift`).
    let spacing = |shift: &str| {
        let lines = format!("    %0: u32 = GetRegister r0\n{shift}");
        let function = parse(&sample_of(&lines, "u32 0x0"));
        dynamic_size_shift(&function, Operand::Value(value(1)))
    };
    let shl = |n: u32| format!("    %1: u32 = ShiftLeftLogical32 %0, u32 {n:#x}\n");
    assert_eq!(
        spacing(&shl(MIN_DYNAMIC_SIZE_SHIFT)),
        MIN_DYNAMIC_SIZE_SHIFT
    );
    assert_eq!(
        spacing(&shl(MIN_DYNAMIC_SIZE_SHIFT - 1)),
        DESCRIPTOR_SIZE_SHIFT
    );
    assert_eq!(spacing(&shl(u32::BITS - 2)), u32::BITS - 2);
    assert_eq!(spacing(&shl(u32::BITS - 1)), DESCRIPTOR_SIZE_SHIFT);
    assert_eq!(
        spacing("    %1: u32 = IMul32 %0, u32 0x10\n"),
        DESCRIPTOR_SIZE_SHIFT
    );
    // Eden's literal values (`size_shift >= 2`, `countr_zero(8)`).
    assert_eq!((MIN_DYNAMIC_SIZE_SHIFT, DESCRIPTOR_SIZE_SHIFT), (2, 3));
    // Through the scan, an index scaled by four bytes gives an array four bytes apart.
    let lines = "    %0: u32 = GetRegister r0\n    %1: u32 = ShiftLeftLogical32 %0, u32 0x2\n    \
                 %2: u32 = GetCbufU32 u32 3, %1\n";
    assert_eq!(
        sources(&sample_of(lines, "%2")),
        "%60: cbuf 3 at 0x0, 1024 handles 4 bytes apart\n"
    );
}

/// The most a handle's trace takes: an OR at the top and two chains reaching the depth limit,
/// `2 * MAX_TRACE_DEPTH + 1` steps (module docs, "Tracing is exact").
#[test]
fn the_longest_trace_takes_the_documented_steps() {
    let side = |first: u32, bank: u32| {
        let mut lines = format!("    %{first}: u32 = GetCbufU32 u32 {bank:#x}, u32 0x10\n");
        for k in 1..MAX_TRACE_DEPTH {
            writeln!(
                lines,
                "    %{}: u32 = BitwiseAnd32 %{}, u32 0xffffffff",
                first + k,
                first + k - 1
            )
            .expect("writes");
        }
        (lines, first + MAX_TRACE_DEPTH - 1)
    };
    let (left, a) = side(0, 3);
    let (right, b) = side(20, 4);
    let lines = format!("{left}{right}    %40: u32 = BitwiseOr32 %{a}, %{b}\n");
    let function = parse(&sample_of(&lines, "%40"));
    let found = scan(&function, buffer(BOUND_BANK)).expect("scans");
    let listed = function.listed().count();
    let steps = u64::from(2 * MAX_TRACE_DEPTH + 1);
    // One block in the block order, every listed instruction, then the trace.
    assert_eq!(
        found.work().steps(),
        1 + u64::try_from(listed).expect("fits") + steps
    );
    // One more operation on either side is past the limit.
    let deeper = format!("{lines}    %41: u32 = BitwiseAnd32 %40, u32 0xff\n");
    assert_eq!(untraced(&sample_of(&deeper, "%41")), Untraced::TooDeep);
}

/// An index word shifted by 32 or more is not kept (Eden's `shift >= 32` check): the array is
/// indexed at run time; 31 is kept.
#[test]
fn an_index_shift_at_the_edge() {
    let indexed = |shift: u32| {
        sample_of(
            &format!(
                "    %0: u32 = GetCbufU32 u32 0x5, u32 0x8\n    \
                 %1: u32 = ShiftLeftLogical32 %0, u32 {shift:#x}\n    \
                 %2: u32 = GetCbufU32 u32 0x3, %1\n"
            ),
            "%2",
        )
    };
    let last = u32::BITS - 1;
    assert_eq!(
        sources(&indexed(last)),
        format!(
            "%60: cbuf 3 at 0x0, 1024 handles 8 bytes apart, index from cbuf 5 at 0x8 << {last}\n"
        )
    );
    assert_eq!(
        sources(&indexed(u32::BITS)),
        "%60: cbuf 3 at 0x0, 1024 handles 8 bytes apart\n"
    );
}
