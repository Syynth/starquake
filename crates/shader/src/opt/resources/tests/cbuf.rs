//! Indexed constant-buffer reads: the bound at its edge for every read width, golden IR, and
//! interpreter equivalence inside the bound.

use super::{Engine, parse, pass};
use crate::info::{INDEXED_CBUF_SIZE, cbuf_element_size};
use crate::interp::{InterpError, Machine};
use crate::ir::{Function, Opcode};
use crate::opt::resources::{run, scan};
use crate::translate::texture::DescriptorBuffer;

/// The bank the reads use: 24, the one the console binds itself (row 65).
const BANK: u32 = 24;

#[test]
fn every_width_divides_the_bound() {
    let widths: Vec<u32> = Opcode::ALL
        .iter()
        .filter_map(|&op| cbuf_element_size(op))
        .collect();
    assert_eq!(widths, [1, 1, 2, 2, 4, 4, 8]);
    for width in widths {
        assert_eq!(INDEXED_CBUF_SIZE % width, 0, "{width}");
    }
    // Eden's indexed size, 64 KiB.
    assert_eq!(INDEXED_CBUF_SIZE, 0x1_0000);
}

/// A read of `opcode` at r0 in bank 24, its bits stored to r1 (and r2 for a pair).
fn read_program(opcode: Opcode) -> String {
    let store = match opcode {
        Opcode::GetCbufF32 => "    %2: u32 = BitCastU32F32 %1\n    %3 = SetRegister r1, %2\n",
        Opcode::GetCbufU32x2 => {
            "    %2: u32 = CompositeExtractU32x2 %1, u32 0x0\n    %3 = SetRegister r1, %2\n    \
             %4: u32 = CompositeExtractU32x2 %1, u32 0x1\n    %5 = SetRegister r2, %4\n"
        }
        _ => "    %3 = SetRegister r1, %1\n",
    };
    format!(
        "block0:\n    %0: u32 = GetRegister r0\n    %1: {} = {opcode} u32 {BANK:#x}, %0\n{store}    return\n",
        opcode.result().name()
    )
}

/// Bank 24 holding all 64 KiB, each byte distinct from its neighbours, r0 = `offset`.
fn machine(offset: u32) -> Machine {
    let mut machine = Machine::new();
    let bytes = (0..INDEXED_CBUF_SIZE)
        .map(|k| u8::try_from((k.wrapping_mul(37) ^ (k >> 8)) & 0xff).expect("a byte"))
        .collect();
    machine.cbufs.insert(BANK, bytes);
    machine.regs[0] = offset;
    machine
}

/// `function` run from r0 = `offset`: r1 and r2, or the error.
fn ran(function: &Function, offset: u32) -> Result<(u32, u32), InterpError> {
    let mut run = machine(offset);
    run.run(function)?;
    Ok((run.regs[1], run.regs[2]))
}

/// The function after the pass, which finds no texture here.
fn bounded(text: &str) -> Function {
    pass(text, &Engine::new()).0
}

const READS: [Opcode; 7] = [
    Opcode::GetCbufU8,
    Opcode::GetCbufS8,
    Opcode::GetCbufU16,
    Opcode::GetCbufS16,
    Opcode::GetCbufU32,
    Opcode::GetCbufF32,
    Opcode::GetCbufU32x2,
];

#[test]
fn reads_at_the_bound_edge() {
    let last = INDEXED_CBUF_SIZE - 1;
    for opcode in READS {
        let text = read_program(opcode);
        let before = parse(&text);
        let after = bounded(&text);
        // Inside the bound, at both ends and between: the read's own value.
        for offset in [0, 1, 0x1234, last - 8, last] {
            assert_eq!(
                ran(&after, offset).expect("runs after"),
                ran(&before, offset).expect("runs before"),
                "{opcode} at {offset:#x}"
            );
        }
        // At the bound and past it: zero, where the unbounded read leaves the bank.
        for offset in [INDEXED_CBUF_SIZE, INDEXED_CBUF_SIZE + 1, u32::MAX] {
            assert_eq!(ran(&after, offset).expect("runs after"), (0, 0), "{opcode}");
            assert!(
                matches!(ran(&before, offset), Err(InterpError::OutOfBounds { .. })),
                "{opcode}"
            );
        }
        // The value read at the last offset is not zero, so the edge case observes the read.
        assert_ne!(ran(&after, last).expect("runs"), (0, 0), "{opcode}");
    }
}

#[test]
fn golden_bounded_reads() {
    let u32_read = bounded(&read_program(Opcode::GetCbufU32));
    insta::assert_snapshot!("bounded_u32", u32_read.to_string());
    let pair = bounded(&read_program(Opcode::GetCbufU32x2));
    insta::assert_snapshot!("bounded_pair", pair.to_string());
    let float = bounded(&read_program(Opcode::GetCbufF32));
    insta::assert_snapshot!("bounded_f32", float.to_string());
}

#[test]
fn immediate_reads_are_left_alone() {
    let text = "block0:
    %0: u32 = GetCbufU32 u32 0x18, u32 0xfffc
    %1: u32x2 = GetCbufU32x2 u32 0x3, u32 0x20000
    %2 = SetRegister r1, %0
    return
";
    assert_eq!(bounded(text), parse(text));
}

/// A read whose bank is a value is bounded the same way: only the offset is checked.
#[test]
fn a_register_bank_read_is_bounded() {
    let text = "block0:
    %0: u32 = GetRegister r0
    %1: u32 = GetRegister r3
    %2: u32 = GetCbufU32 %1, %0
    %3 = SetRegister r1, %2
    return
";
    let after = bounded(text);
    insta::assert_snapshot!("bounded_register_bank", after.to_string());
    let mut start = machine(INDEXED_CBUF_SIZE);
    start.regs[3] = BANK;
    start.run(&after).expect("runs");
    assert_eq!(start.regs[1], 0);
}

/// Every read the blocks list is bounded, in every block, reachable or not.
#[test]
fn reads_in_every_block_are_bounded() {
    let text = "block0:
    %0: u32 = GetRegister r0
    %1: u32 = GetCbufU32 u32 0x18, %0
    return
block1:
    %2: u32 = GetCbufU16 u32 0x18, %0
    return
";
    let mut function = parse(text);
    let found = scan(&function, DescriptorBuffer::new()).expect("scans");
    let (table, work) = run(&mut function, &found, &crate::exchange::Answers::new()).expect("runs");
    assert!(table.is_empty());
    let reads = function
        .listed()
        .filter(|(_, inst)| {
            matches!(inst, crate::ir::Inst::Op(op) if cbuf_element_size(op.opcode).is_some())
        })
        .count();
    let selects = function
        .listed()
        .filter(
            |(_, inst)| matches!(inst, crate::ir::Inst::Op(op) if op.opcode == Opcode::SelectU32),
        )
        .count();
    assert_eq!((reads, selects), (2, 4));
    // Three listed instructions walked, then the two blocks' original instructions placed.
    assert_eq!(work.steps(), 3 + 3);
}
