//! Memory spaces and the rest of the context: constant buffers, global, storage, shared and
//! local memory with their alignment and bounds checks, the integer atomics, attributes,
//! patches, fragment outputs, registers, predicates, flags, variables and system values.

use super::{eval_error_on, run};
use crate::interp::{Machine, Region, Space, Val};
use crate::ir::{Attribute, Patch};

/// The value of `%id` after running the straight-line `body` on `machine`.
fn value_on(machine: &mut Machine, body: &str, id: u32) -> Val {
    let text = format!("block0:\n{body}\n    return\n");
    run(&text, machine)
        .expect("runs")
        .value(id)
        .expect("a value")
}

/// The value of the single operation `op` on `machine`.
fn op_on(machine: &mut Machine, op: &str) -> Val {
    value_on(machine, &format!("    %0 = {op}"), 0)
}

/// `len` distinct bytes: byte `i` is `(i + 1) * 0x11` modulo 256 (0x11, 0x22, ... 0xff, 0x10).
fn bytes(len: u8) -> Vec<u8> {
    (1..=len).map(|index| index.wrapping_mul(0x11)).collect()
}

#[test]
fn constant_buffers_read_at_the_aligned_offset() {
    let mut machine = Machine::new();
    machine.cbufs.insert(1, bytes(16));
    machine.cbufs.insert(2, vec![0xaa; 16]);
    let cases = [
        ("GetCbufU32 u32 1, u32 4", Val::U32(0x8877_6655)),
        // Eden: "Hardware been proved to read the aligned offset".
        ("GetCbufU32 u32 1, u32 6", Val::U32(0x8877_6655)),
        ("GetCbufU32 u32 2, u32 4", Val::U32(0xaaaa_aaaa)),
        ("GetCbufF32 u32 1, u32 0", Val::F32(0x4433_2211)),
        ("GetCbufU8 u32 1, u32 3", Val::U32(0x44)),
        ("GetCbufS8 u32 1, u32 7", Val::U32(0xffff_ff88)),
        ("GetCbufU8 u32 1, u32 7", Val::U32(0x88)),
        ("GetCbufS8 u32 1, u32 6", Val::U32(0x77)),
        ("GetCbufU16 u32 1, u32 3", Val::U32(0x4433)),
        ("GetCbufS16 u32 1, u32 7", Val::U32(0xffff_8877)),
        ("GetCbufU16 u32 1, u32 7", Val::U32(0x8877)),
        (
            "GetCbufU32x2 u32 1, u32 8",
            Val::U32x2([0xccbb_aa99, 0x10ff_eedd]),
        ),
        (
            "GetCbufU32x2 u32 1, u32 12",
            Val::U32x2([0xccbb_aa99, 0x10ff_eedd]),
        ),
    ];
    for (op, want) in cases {
        assert_eq!(op_on(&mut machine, op), want, "{op}");
    }
}

#[test]
fn constant_buffer_bounds() {
    let mut machine = Machine::new();
    machine.cbufs.insert(1, bytes(8));
    // The last word ends exactly at the end; the next starts there.
    assert_eq!(
        op_on(&mut machine, "GetCbufU32 u32 1, u32 4"),
        Val::U32(0x8877_6655)
    );
    assert_eq!(
        eval_error_on("GetCbufU32 u32 1, u32 8", &mut machine),
        "%0: an access of 4 bytes to constant buffer 1 at 0x8 is outside the 8 bytes it holds"
    );
    assert_eq!(
        op_on(&mut machine, "GetCbufU32x2 u32 1, u32 4"),
        Val::U32x2([0x4433_2211, 0x8877_6655])
    );
    assert_eq!(
        eval_error_on("GetCbufU32x2 u32 1, u32 8", &mut machine),
        "%0: an access of 8 bytes to constant buffer 1 at 0x8 is outside the 8 bytes it holds"
    );
    assert_eq!(
        eval_error_on("GetCbufU32 u32 3, u32 0", &mut machine),
        "%0: an access of 4 bytes to constant buffer 3 at 0x0 is outside the 0 bytes it holds"
    );
    assert_eq!(
        eval_error_on("GetCbufU32 u32 1, u32 0xffffffff", &mut machine),
        "%0: an access of 4 bytes to constant buffer 1 at 0xfffffffc is outside the 8 bytes it holds"
    );
}

/// A machine with 16 bytes of global memory at 0x1000.
fn global() -> Machine {
    let mut machine = Machine::new();
    machine.global = Region {
        base: 0x1000,
        bytes: bytes(16),
    };
    machine
}

#[test]
fn global_loads_by_width() {
    let mut machine = global();
    let cases = [
        ("LoadGlobalU8 u64 0x1007", Val::U32(0x88)),
        ("LoadGlobalS8 u64 0x1007", Val::U32(0xffff_ff88)),
        ("LoadGlobalS8 u64 0x1006", Val::U32(0x77)),
        ("LoadGlobalU16 u64 0x1006", Val::U32(0x8877)),
        ("LoadGlobalS16 u64 0x1006", Val::U32(0xffff_8877)),
        ("LoadGlobalS16 u64 0x1004", Val::U32(0x6655)),
        ("LoadGlobal32 u64 0x100c", Val::U32(0x10ff_eedd)),
        (
            "LoadGlobal64 u64 0x1008",
            Val::U32x2([0xccbb_aa99, 0x10ff_eedd]),
        ),
        (
            "LoadGlobal128 u64 0x1000",
            Val::U32x4([0x4433_2211, 0x8877_6655, 0xccbb_aa99, 0x10ff_eedd]),
        ),
    ];
    for (op, want) in cases {
        assert_eq!(op_on(&mut machine, op), want, "{op}");
    }
}

#[test]
fn global_alignment_then_bounds() {
    let mut machine = global();
    let err = |op: &str, machine: &mut Machine| eval_error_on(op, machine);
    assert_eq!(
        err("LoadGlobal32 u64 0x1010", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x1010 is outside the 16 bytes it holds"
    );
    assert_eq!(
        err("LoadGlobal32 u64 0xffc", &mut machine),
        "%0: an access of 4 bytes to global memory at 0xffc is outside the 16 bytes it holds"
    );
    assert_eq!(
        err("LoadGlobal32 u64 0x1002", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x1002 is not aligned"
    );
    // Both faults: the alignment check comes first.
    assert_eq!(
        err("LoadGlobal32 u64 0x2002", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x2002 is not aligned"
    );
    assert_eq!(
        err("LoadGlobal128 u64 0x1008", &mut machine),
        "%0: an access of 16 bytes to global memory at 0x1008 is not aligned"
    );
    // An access whose end overflows the address space.
    assert_eq!(
        err("LoadGlobal64 u64 0xfffffffffffffff8", &mut machine),
        "%0: an access of 8 bytes to global memory at 0xfffffffffffffff8 is outside the 16 bytes it holds"
    );
    machine.global.base = 0xffff_ffff_ffff_fff8;
    assert_eq!(
        op_on(&mut machine, "LoadGlobal64 u64 0xfffffffffffffff8"),
        Val::U32x2([0x4433_2211, 0x8877_6655])
    );
    assert_eq!(
        err("LoadGlobal32 u64 0x0", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x0 is outside the 16 bytes it holds"
    );
}

#[test]
fn global_stores_write_only_their_bytes() {
    let mut machine = global();
    let body = "    %0 = WriteGlobalU8 u64 0x1001, u32 0xabcdef01
    %1 = WriteGlobalS16 u64 0x1006, u32 0x12345678
    %2 = CompositeConstructU32x2 u32 0xa0a1a2a3, u32 0xb0b1b2b3
    %3 = WriteGlobal64 u64 0x1008, %2";
    run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.global.bytes,
        [
            0x11, 0x01, 0x33, 0x44, 0x55, 0x66, 0x78, 0x56, 0xa3, 0xa2, 0xa1, 0xa0, 0xb3, 0xb2,
            0xb1, 0xb0
        ]
    );
    let mut machine = global();
    let wide = "    %0 = CompositeConstructU32x4 u32 0x1, u32 0x2, u32 0x3, u32 0x4
    %1 = WriteGlobal128 u64 0x1000, %0
    %2 = WriteGlobal32 u64 0x1004, u32 0x05060708
    %3 = WriteGlobalU16 u64 0x100e, u32 0xffff1234
    %4 = WriteGlobalS8 u64 0x100c, u32 0x99";
    run(&format!("block0:\n{wide}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.global.bytes,
        [1, 0, 0, 0, 8, 7, 6, 5, 3, 0, 0, 0, 0x99, 0, 0x34, 0x12]
    );
    // A refused store leaves the memory as it was.
    let before = machine.global.bytes.clone();
    assert_eq!(
        eval_error_on("WriteGlobal32 u64 0x100e, u32 0x0", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x100e is not aligned"
    );
    assert_eq!(
        eval_error_on("WriteGlobal32 u64 0x1010, u32 0x0", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x1010 is outside the 16 bytes it holds"
    );
    assert_eq!(machine.global.bytes, before);
}

#[test]
fn storage_buffers_by_binding() {
    let mut machine = Machine::new();
    machine.storage.insert(2, bytes(8));
    assert_eq!(
        op_on(&mut machine, "LoadStorage32 u32 2, u32 4"),
        Val::U32(0x8877_6655)
    );
    assert_eq!(
        op_on(&mut machine, "LoadStorageS8 u32 2, u32 4"),
        Val::U32(0x55)
    );
    assert_eq!(
        op_on(&mut machine, "LoadStorageS16 u32 2, u32 6"),
        Val::U32(0xffff_8877)
    );
    assert_eq!(
        op_on(&mut machine, "LoadStorageU16 u32 2, u32 6"),
        Val::U32(0x8877)
    );
    assert_eq!(
        op_on(&mut machine, "LoadStorageU8 u32 2, u32 7"),
        Val::U32(0x88)
    );
    assert_eq!(
        op_on(&mut machine, "LoadStorage64 u32 2, u32 0"),
        Val::U32x2([0x4433_2211, 0x8877_6655])
    );
    assert_eq!(
        eval_error_on("LoadStorage128 u32 2, u32 0", &mut machine),
        "%0: an access of 16 bytes to storage buffer 2 at 0x0 is outside the 8 bytes it holds"
    );
    assert_eq!(
        eval_error_on("LoadStorage32 u32 3, u32 0", &mut machine),
        "%0: an access of 4 bytes to storage buffer 3 at 0x0 is outside the 0 bytes it holds"
    );
    assert_eq!(
        eval_error_on("WriteStorage32 u32 3, u32 0, u32 0x1", &mut machine),
        "%0: an access of 4 bytes to storage buffer 3 at 0x0 is outside the 0 bytes it holds"
    );
    let body = "    %0 = WriteStorageU8 u32 2, u32 1, u32 0xfff
    %1 = WriteStorage32 u32 2, u32 4, u32 0xdeadbeef
    %2 = WriteStorageS16 u32 2, u32 2, u32 0xabcd
    %3 = CompositeConstructU32x2 u32 0x1, u32 0x2";
    run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.storage.get(&2).map(Vec::as_slice),
        Some(&[0x11, 0xff, 0xcd, 0xab, 0xef, 0xbe, 0xad, 0xde][..])
    );
    let wide = "    %0 = CompositeConstructU32x2 u32 0x1, u32 0x2
    %1 = WriteStorage64 u32 2, u32 0, %0";
    run(&format!("block0:\n{wide}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.storage.get(&2).map(Vec::as_slice),
        Some(&[1, 0, 0, 0, 2, 0, 0, 0][..])
    );
}

#[test]
fn shared_memory() {
    let mut machine = Machine::new();
    machine.shared = bytes(16);
    let cases = [
        ("LoadSharedU8 u32 0x7", Val::U32(0x88)),
        ("LoadSharedS8 u32 0x7", Val::U32(0xffff_ff88)),
        ("LoadSharedU16 u32 0x6", Val::U32(0x8877)),
        ("LoadSharedS16 u32 0x6", Val::U32(0xffff_8877)),
        ("LoadSharedU32 u32 0x4", Val::U32(0x8877_6655)),
        (
            "LoadSharedU64 u32 0x8",
            Val::U32x2([0xccbb_aa99, 0x10ff_eedd]),
        ),
        (
            "LoadSharedU128 u32 0x0",
            Val::U32x4([0x4433_2211, 0x8877_6655, 0xccbb_aa99, 0x10ff_eedd]),
        ),
    ];
    for (op, want) in cases {
        assert_eq!(op_on(&mut machine, op), want, "{op}");
    }
    assert_eq!(
        eval_error_on("LoadSharedU32 u32 0x10", &mut machine),
        "%0: an access of 4 bytes to shared memory at 0x10 is outside the 16 bytes it holds"
    );
    assert_eq!(
        eval_error_on("LoadSharedU64 u32 0x4", &mut machine),
        "%0: an access of 8 bytes to shared memory at 0x4 is not aligned"
    );
    let body = "    %0 = WriteSharedU8 u32 0x0, u32 0x1ff
    %1 = WriteSharedU16 u32 0x2, u32 0x10203
    %2 = WriteSharedU32 u32 0x4, u32 0x0
    %3 = CompositeConstructU32x2 u32 0x1, u32 0x2
    %4 = WriteSharedU64 u32 0x8, %3";
    run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.shared,
        [0xff, 0x22, 0x03, 0x02, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
    );
    let wide = "    %0 = CompositeConstructU32x4 u32 0x1, u32 0x2, u32 0x3, u32 0x4
    %1 = WriteSharedU128 u32 0x0, %0";
    run(&format!("block0:\n{wide}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(
        machine.shared,
        [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
    );
}

#[test]
fn error_words_count_bytes_in_the_singular_and_plural() {
    let mut machine = Machine::new();
    machine.shared = vec![0x80];
    assert_eq!(
        op_on(&mut machine, "LoadSharedS8 u32 0x0"),
        Val::U32(0xffff_ff80)
    );
    assert_eq!(
        eval_error_on("LoadSharedU8 u32 0x1", &mut machine),
        "%0: an access of 1 byte to shared memory at 0x1 is outside the 1 byte it holds"
    );
    machine.shared.clear();
    assert_eq!(
        eval_error_on("LoadSharedU8 u32 0x0", &mut machine),
        "%0: an access of 1 byte to shared memory at 0x0 is outside the 0 bytes it holds"
    );
}

#[test]
fn local_memory_by_word() {
    let mut machine = Machine::new();
    machine.local = vec![10, 20, 30];
    assert_eq!(op_on(&mut machine, "LoadLocal u32 2"), Val::U32(30));
    assert_eq!(
        eval_error_on("LoadLocal u32 3", &mut machine),
        "%0: an access of 4 bytes to local memory at 0xc is outside the 12 bytes it holds"
    );
    assert_eq!(
        eval_error_on("WriteLocal u32 0xffffffff, u32 1", &mut machine),
        "%0: an access of 4 bytes to local memory at 0x3fffffffc is outside the 12 bytes it holds"
    );
    run(
        "block0:\n    %0 = WriteLocal u32 1, u32 99\n    return\n",
        &mut machine,
    )
    .expect("runs");
    assert_eq!(machine.local, [10, 99, 30]);
}

/// Runs `op` on shared memory word 0 holding `old`; returns the result and the new word.
fn shared_atomic(op: &str, old: u32, value: u32) -> (u32, u32) {
    let mut machine = Machine::new();
    machine.shared = old.to_le_bytes().to_vec();
    let result = op_on(&mut machine, &format!("{op} u32 0x0, u32 {value:#x}"));
    let mut word = [0; 4];
    word.copy_from_slice(&machine.shared);
    (super::u32_of(result), u32::from_le_bytes(word))
}

#[test]
fn atomics_return_the_old_value_and_store_the_new() {
    // 0xfffffffe is -2 signed and large unsigned.
    let old = 0xffff_fffe;
    let cases = [
        ("SharedAtomicIAdd32", old, 3, 1),
        ("SharedAtomicSMin32", old, 1, old),
        ("SharedAtomicUMin32", old, 1, 1),
        ("SharedAtomicSMax32", old, 1, 1),
        ("SharedAtomicUMax32", old, 1, old),
        ("SharedAtomicAnd32", 0xf0f0, 0xff00, 0xf000),
        ("SharedAtomicOr32", 0xf0f0, 0xff00, 0xfff0),
        ("SharedAtomicXor32", 0xf0f0, 0xff00, 0x0ff0),
        ("SharedAtomicExchange32", 7, 9, 9),
        // Increment wraps to 0 when old >= value.
        ("SharedAtomicInc32", 5, 5, 0),
        ("SharedAtomicInc32", 6, 5, 0),
        ("SharedAtomicInc32", 4, 5, 5),
        // Decrement wraps to value when old is 0 or above value.
        ("SharedAtomicDec32", 0, 5, 5),
        ("SharedAtomicDec32", 6, 5, 5),
        ("SharedAtomicDec32", 5, 5, 4),
        ("SharedAtomicDec32", 1, 5, 0),
    ];
    for (op, old, value, new) in cases {
        assert_eq!(
            shared_atomic(op, old, value),
            (old, new),
            "{op} {old:#x} {value:#x}"
        );
    }
}

#[test]
fn global_and_storage_atomics() {
    let mut machine = global();
    assert_eq!(
        op_on(&mut machine, "GlobalAtomicIAdd32 u64 0x1004, u32 0x1"),
        Val::U32(0x8877_6655)
    );
    assert_eq!(
        machine.global.bytes.get(4..8),
        Some(&[0x56, 0x66, 0x77, 0x88][..])
    );
    assert_eq!(
        eval_error_on("GlobalAtomicExchange32 u64 0x1002, u32 0x1", &mut machine),
        "%0: an access of 4 bytes to global memory at 0x1002 is not aligned"
    );
    let mut machine = Machine::new();
    machine.storage.insert(5, vec![0, 0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(
        op_on(
            &mut machine,
            "StorageAtomicExchange32 u32 5, u32 4, u32 0x7"
        ),
        Val::U32(1)
    );
    assert_eq!(
        machine.storage.get(&5).map(Vec::as_slice),
        Some(&[0, 0, 0, 0, 7, 0, 0, 0][..])
    );
    assert_eq!(
        eval_error_on("StorageAtomicIAdd32 u32 5, u32 8, u32 0x7", &mut machine),
        "%0: an access of 4 bytes to storage buffer 5 at 0x8 is outside the 8 bytes it holds"
    );
}

#[test]
fn attributes_by_word_and_vertex() {
    let mut machine = Machine::new();
    machine.attributes.insert((Attribute(0x20), 0), 0x3f80_0000);
    machine.attributes.insert((Attribute(0x20), 1), 0x4000_0000);
    machine.attributes.insert((Attribute(0x21), 0), 7);
    let body = "    %0 = GetAttribute a0x80, u32 0x0
    %1 = GetAttribute a0x80, u32 0x1
    %2 = GetAttributeU32 a0x84, u32 0x0
    %3 = GetAttributeIndexed u32 0x86, u32 0x0
    %4 = SetAttribute a0x70, %0, u32 0x0
    %5 = SetAttributeIndexed u32 0x74, %1, u32 0x2";
    assert_eq!(value_on(&mut machine, body, 0), Val::F32(0x3f80_0000));
    assert_eq!(value_on(&mut machine, body, 1), Val::F32(0x4000_0000));
    assert_eq!(value_on(&mut machine, body, 2), Val::U32(7));
    assert_eq!(value_on(&mut machine, body, 3), Val::F32(7));
    assert_eq!(
        machine
            .outputs
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect::<Vec<_>>(),
        [
            ((Attribute(0x1c), 0), 0x3f80_0000),
            ((Attribute(0x1d), 2), 0x4000_0000)
        ]
    );
    assert_eq!(
        eval_error_on("GetAttribute a0x84, u32 0x1", &mut machine),
        "%0: attribute a0x84 of vertex 1 is not given"
    );
    // The largest offset that names a word, and the first that does not.
    assert_eq!(
        eval_error_on("GetAttributeIndexed u32 0x3ffff, u32 0x0", &mut machine),
        "%0: attribute a0x3fffc of vertex 0 is not given"
    );
    assert_eq!(
        eval_error_on("GetAttributeIndexed u32 0x40000, u32 0x0", &mut machine),
        "%0: attribute offset 0x40000 names no attribute"
    );
    assert_eq!(
        eval_error_on(
            "SetAttributeIndexed u32 0x40000, f32 1.0, u32 0x0",
            &mut machine
        ),
        "%0: attribute offset 0x40000 names no attribute"
    );
}

#[test]
fn patches_and_fragment_outputs() {
    let mut machine = Machine::new();
    let patch = |index| Patch::new(index).expect("a patch");
    machine.patches.insert(patch(7), 0x4040_0000);
    let body = "    %0 = GetPatch patch7
    %1 = SetPatch patch8, %0
    %2 = GetPatch patch8
    %3 = SetFragColor u32 0x1, u32 0x3, f32 0.5
    %4 = SetFragDepth f32 0.25
    %5 = SetSampleMask u32 0xf0";
    assert_eq!(value_on(&mut machine, body, 2), Val::F32(0x4040_0000));
    assert_eq!(machine.patches.get(&patch(8)), Some(&0x4040_0000));
    assert_eq!(
        machine
            .frag_colors
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect::<Vec<_>>(),
        [((1, 3), 0x3f00_0000)]
    );
    assert_eq!(machine.frag_depth, Some(0x3e80_0000));
    assert_eq!(machine.sample_mask, Some(0xf0));
    assert_eq!(
        eval_error_on("GetPatch patch9", &mut machine),
        "%0: patch9 is not given"
    );
}

#[test]
fn registers_predicates_flags_and_variables() {
    let mut machine = Machine::new();
    machine.regs[3] = 0x1234;
    machine.regs[254] = 9;
    machine.preds[6] = true;
    machine.uniform_regs[62] = 0x55;
    machine.uniform_preds[2] = true;
    let body = "    %0 = GetRegister r3
    %1 = IAdd32 %0, u32 0x1
    %2 = SetRegister r4, %1
    %3 = GetRegister r254
    %4 = GetPred p6
    %5 = LogicalNot %4
    %6 = SetPred p0, %5
    %7 = SetPred p1, %4
    %8 = GetUniformRegister ur62
    %9 = SetUniformRegister ur0, %8
    %10 = GetUniformPred up2
    %11 = SetUniformPred up3, %10
    %12 = GetGotoVariable u32 0x5
    %13 = SetGotoVariable u32 0x5, u1 true
    %14 = GetGotoVariable u32 0x5
    %15 = GetIndirectBranchVariable
    %16 = SetIndirectBranchVariable u32 0x40
    %17 = GetIndirectBranchVariable
    %18 = GetZFlag
    %19 = SetZFlag u1 true
    %20 = SetSFlag u1 true
    %21 = SetCFlag u1 true
    %22 = SetOFlag u1 true
    %23 = GetOFlag
    %24 = GetSFlag
    %25 = GetCFlag
    %26 = GetZFlag";
    let finished = run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(machine.regs[4], 0x1235);
    assert_eq!(finished.value(3), Some(Val::U32(9)));
    assert_eq!((machine.preds[0], machine.preds[1]), (false, true));
    assert_eq!(machine.uniform_regs[0], 0x55);
    assert!(machine.uniform_preds[3]);
    assert_eq!(finished.value(12), Some(Val::U1(false)));
    assert_eq!(finished.value(14), Some(Val::U1(true)));
    assert_eq!(finished.value(15), Some(Val::U32(0)));
    assert_eq!(finished.value(17), Some(Val::U32(0x40)));
    assert_eq!(machine.indirect_branch, 0x40);
    assert_eq!(finished.value(18), Some(Val::U1(false)));
    for id in 23..=26 {
        assert_eq!(finished.value(id), Some(Val::U1(true)), "%{id}");
    }
    assert_eq!(machine.goto_variables.get(&5), Some(&true));
    let flags = machine.flags;
    assert!(flags.zero && flags.sign && flags.carry && flags.overflow);
}

#[test]
fn flags_are_read_independently() {
    // Each flag set alone is read by its own opcode only.
    let opcodes = ["Z", "S", "C", "O"];
    for (set, setter) in opcodes.iter().enumerate() {
        let mut lines = vec![format!("    %0 = Set{setter}Flag u1 true")];
        for (index, getter) in opcodes.iter().enumerate() {
            lines.push(format!("    %{} = Get{getter}Flag", index + 1));
        }
        let body = lines.join("\n");
        for (index, _) in opcodes.iter().enumerate() {
            let id = u32::try_from(index + 1).expect("small");
            assert_eq!(
                value_on(&mut Machine::new(), &body, id),
                Val::U1(index == set),
                "set {setter}"
            );
        }
    }
}

#[test]
fn system_values() {
    let mut machine = Machine::new();
    assert_eq!(op_on(&mut machine, "YDirection"), Val::F32(0x3f80_0000));
    machine.system.workgroup_id = [1, 2, 3];
    machine.system.local_invocation_id = [4, 5, 6];
    machine.system.invocation_id = 7;
    machine.system.sample_id = 8;
    machine.system.lane_id = 9;
    machine.system.y_direction = 0xbf80_0000;
    assert_eq!(op_on(&mut machine, "WorkgroupId"), Val::U32x3([1, 2, 3]));
    assert_eq!(
        op_on(&mut machine, "LocalInvocationId"),
        Val::U32x3([4, 5, 6])
    );
    assert_eq!(op_on(&mut machine, "InvocationId"), Val::U32(7));
    assert_eq!(op_on(&mut machine, "SampleId"), Val::U32(8));
    assert_eq!(op_on(&mut machine, "LaneId"), Val::U32(9));
    assert_eq!(op_on(&mut machine, "YDirection"), Val::F32(0xbf80_0000));
    assert_eq!(op_on(&mut machine, "IsHelperInvocation"), Val::U1(false));
    machine.system.is_helper = true;
    assert_eq!(op_on(&mut machine, "IsHelperInvocation"), Val::U1(true));
    // Demoting makes the invocation a helper and lets it continue.
    let mut machine = Machine::new();
    let body = "    %0 = IsHelperInvocation
    %1 = DemoteToHelperInvocation
    %2 = IsHelperInvocation
    %3 = SetRegister r0, u32 0x1";
    let finished = run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(finished.value(0), Some(Val::U1(false)));
    assert_eq!(finished.value(2), Some(Val::U1(true)));
    assert!(machine.demoted);
    assert_eq!(machine.regs[0], 1);
}

#[test]
fn no_ops_leave_the_machine_unchanged() {
    let mut machine = Machine::new();
    machine.regs[1] = 5;
    let before = machine.clone();
    let body = "    %0 = LaneId
    %1 = Reference %0
    %2 = Prologue
    %3 = Epilogue
    %4 = Barrier
    %5 = WorkgroupMemoryBarrier
    %6 = DeviceMemoryBarrier";
    run(&format!("block0:\n{body}\n    return\n"), &mut machine).expect("runs");
    assert_eq!(machine, before);
}

#[test]
fn memory_error_spaces_print_in_words() {
    let cases = [
        (Space::ConstantBuffer(2), "constant buffer 2"),
        (Space::Global, "global memory"),
        (Space::Storage(4), "storage buffer 4"),
        (Space::Local, "local memory"),
        (Space::Shared, "shared memory"),
    ];
    for (space, words) in cases {
        assert_eq!(space.to_string(), words);
    }
}
