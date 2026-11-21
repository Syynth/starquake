//! The pass's tests: golden IR and tables, each refusal with its exact reason, the search
//! against a direct port of Eden's breadth-first search, interpreter equivalence (hand-written
//! IR and programs from the translator), and the work bound at the translation driver's cap.

use super::*;
use crate::interp::{Machine, Region};
use crate::ir::Type;

/// Parses `text`, which must be well-formed.
fn parse(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// The alignment of `bytes`, a power of two.
fn align(bytes: u32) -> StorageAlignment {
    StorageAlignment::new(bytes).expect("a power of two")
}

/// The host alignment most tests use: 64 bytes, so a pointer's low 6 bits are cleared.
const ALIGN: u32 = 64;

/// The text of `text` after the pass, with the table.
fn convert(text: &str) -> (String, StorageBuffers) {
    let mut function = parse(text);
    let buffers = to_storage_buffers(&mut function, align(ALIGN)).expect("converts");
    (function.to_string(), buffers)
}

/// The error the pass gives on `text`, after checking it left the function unchanged.
fn refuse(text: &str) -> GlobalMemoryError {
    let mut function = parse(text);
    let before = function.clone();
    let error = to_storage_buffers(&mut function, align(ALIGN)).expect_err("refuses");
    assert_eq!(function, before, "a refused function is unchanged");
    error
}

/// The value id `index`.
fn value(index: u32) -> Value {
    let text = format!("block0:\n    %{index} = Void\n    return\n");
    parse(&text)
        .blocks()
        .next()
        .and_then(|(_, block)| block.insts.first().copied())
        .expect("one instruction")
}

/// A descriptor, as tests expect one.
const fn descriptor(bank: u8, offset: u32, written: bool) -> StorageBufferDescriptor {
    StorageBufferDescriptor {
        bank,
        offset,
        written,
    }
}

/// A load through a biased pointer (bank 0, `0x110`, read as a pair) at `+0x10`, a store and a
/// reduction through an unbiased one (bank 3, `0x8`), and an atomic whose old value is output
/// through a third (bank 1, `0x40`).
const MIXED: &str = "block0:
    %0: u32x2 = GetCbufU32x2 u32 0x0, u32 0x110
    %1: u32 = CompositeExtractU32x2 %0, u32 0x0
    %2: u32 = CompositeExtractU32x2 %0, u32 0x1
    %3: u32x2 = CompositeConstructU32x2 %1, %2
    %4: u64 = PackUint2x32 %3
    %5: u64 = IAdd64 %4, u64 0x10
    %6: u32 = LoadGlobal32 %5
    %7: u32 = GetCbufU32 u32 0x3, u32 0x8
    %8: u32 = GetCbufU32 u32 0x3, u32 0xc
    %9: u32x2 = CompositeConstructU32x2 %7, %8
    %10: u64 = PackUint2x32 %9
    %11 = WriteGlobal32 %10, %6
    %12: u32 = GetCbufU32 u32 0x1, u32 0x40
    %13: u32 = GetCbufU32 u32 0x1, u32 0x44
    %14: u32x2 = CompositeConstructU32x2 %12, %13
    %15: u64 = PackUint2x32 %14
    %16: u32 = GlobalAtomicIAdd32 %15, %6
    %17: u64 = IAdd64 %10, u64 0x4
    %18: u32 = GlobalAtomicUMax32 %17, u32 0x5
    %19: f32 = BitCastF32U32 %16
    %20 = SetAttribute a0x80, %19, u32 0x0
    return
";

mod golden {
    use super::*;

    #[test]
    fn loads_stores_atomics_and_reductions_convert() {
        let (text, buffers) = convert(MIXED);
        assert_eq!(
            text,
            "block0:
    %0: u32x2 = GetCbufU32x2 u32 0x0, u32 0x110
    %1: u32 = CompositeExtractU32x2 %0, u32 0x0
    %2: u32 = CompositeExtractU32x2 %0, u32 0x1
    %3: u32x2 = CompositeConstructU32x2 %1, %2
    %4: u64 = PackUint2x32 %3
    %5: u64 = IAdd64 %4, u64 0x10
    %21: u32 = IAdd32 %1, u32 0x10
    %22: u32 = GetCbufU32 u32 0x0, u32 0x110
    %23: u32 = BitwiseAnd32 %22, u32 0xffffffc0
    %24: u32 = ISub32 %21, %23
    %25: u32 = LoadStorage32 u32 0x0, %24
    %6: opaque = Identity %25
    %7: u32 = GetCbufU32 u32 0x3, u32 0x8
    %8: u32 = GetCbufU32 u32 0x3, u32 0xc
    %9: u32x2 = CompositeConstructU32x2 %7, %8
    %10: u64 = PackUint2x32 %9
    %26: u32 = GetCbufU32 u32 0x3, u32 0x8
    %27: u32 = BitwiseAnd32 %26, u32 0xffffffc0
    %28: u32 = ISub32 %7, %27
    %29 = WriteStorage32 u32 0x2, %28, %6
    %11 = Void
    %12: u32 = GetCbufU32 u32 0x1, u32 0x40
    %13: u32 = GetCbufU32 u32 0x1, u32 0x44
    %14: u32x2 = CompositeConstructU32x2 %12, %13
    %15: u64 = PackUint2x32 %14
    %30: u32 = GetCbufU32 u32 0x1, u32 0x40
    %31: u32 = BitwiseAnd32 %30, u32 0xffffffc0
    %32: u32 = ISub32 %12, %31
    %33: u32 = StorageAtomicIAdd32 u32 0x1, %32, %6
    %16: opaque = Identity %33
    %17: u64 = IAdd64 %10, u64 0x4
    %34: u32 = IAdd32 %7, u32 0x4
    %35: u32 = GetCbufU32 u32 0x3, u32 0x8
    %36: u32 = BitwiseAnd32 %35, u32 0xffffffc0
    %37: u32 = ISub32 %34, %36
    %38: u32 = StorageAtomicUMax32 u32 0x2, %37, u32 0x5
    %18: opaque = Identity %38
    %19: f32 = BitCastF32U32 %16
    %20 = SetAttribute a0x80, %19, u32 0x0
    return
"
        );
        // Sorted by bank, not by first use; the store and the reduction share bank 3's
        // buffer; only bank 0's is read and never written.
        assert_eq!(
            buffers.descriptors(),
            [
                descriptor(0, 0x110, false),
                descriptor(1, 0x40, true),
                descriptor(3, 0x8, true),
            ]
        );
    }

    /// The exact work of [`MIXED`] (module docs, "Cost"): 1 for the block order, 21 for the
    /// scan of its listed instructions, 4 for the low addresses, 30 for the search's set-up (4
    /// instructions reached and their 8 arguments, 1 argument edge counted and 1 filled, then 7
    /// steps of the biased walk and 9 of the unbiased one), 10 for the checks (4 accesses and 6
    /// steps of finding), 21 for rebuilding the block and 18 for the instructions made.
    #[test]
    fn the_work_of_a_fixed_function_is_pinned() {
        let (_, buffers) = convert(MIXED);
        assert_eq!(buffers.work().steps(), WORK_OF_MIXED);
    }

    /// The pinned work of [`MIXED`] (`the_work_of_a_fixed_function_is_pinned`).
    const WORK_OF_MIXED: u64 = 105;

    /// Every width: the 8-, 16-, 64- and 128-bit loads and stores keep their data argument
    /// and type.
    #[test]
    fn every_width_keeps_its_type() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x2, u32 0x18
    %1: u32 = GetCbufU32 u32 0x2, u32 0x1c
    %2: u32x2 = CompositeConstructU32x2 %0, %1
    %3: u64 = PackUint2x32 %2
    %4: u32 = LoadGlobalS8 %3
    %5: u32x2 = LoadGlobal64 %3
    %6: u32x4 = LoadGlobal128 %3
    %7 = WriteGlobalU16 %3, %4
    %8 = WriteGlobal64 %3, %5
    %9 = WriteGlobal128 %3, %6
    return
";
        let (out, buffers) = convert(text);
        for line in [
            "%11: u32 = BitwiseAnd32 %10, u32 0xffffffc0",
            "%13: u32 = LoadStorageS8 u32 0x0, %12",
            "%17: u32x2 = LoadStorage64 u32 0x0, %16",
            "%21: u32x4 = LoadStorage128 u32 0x0, %20",
            "%25 = WriteStorageU16 u32 0x0, %24, %4",
            "%29 = WriteStorage64 u32 0x0, %28, %5",
            "%33 = WriteStorage128 u32 0x0, %32, %6",
        ] {
            assert!(out.contains(line), "{line} in\n{out}");
        }
        assert_eq!(buffers.descriptors(), [descriptor(2, 0x18, true)]);
    }

    /// A negative immediate offset keeps its low word; one whose low word is zero adds
    /// nothing, as Eden reads the offset as an `s32` (module docs, "Ported as Eden has it").
    #[test]
    fn an_offset_with_a_zero_low_word_adds_nothing() {
        let access = |offset: u64| {
            format!(
                "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x100
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u64 = IAdd64 %2, u64 {offset:#x}
    %4: u32 = LoadGlobal32 %3
    return
"
            )
        };
        let (negative, _) = convert(&access(0xffff_ffff_ffff_fff0));
        assert!(
            negative.contains("%5: u32 = IAdd32 %0, u32 0xfffffff0"),
            "{negative}"
        );
        let (zero_low, _) = convert(&access(0x1_0000_0000));
        assert!(
            zero_low.contains("%5: u32 = GetCbufU32 u32 0x0, u32 0x100"),
            "{zero_low}"
        );
        assert!(!zero_low.contains("IAdd32"), "{zero_low}");
        // The smallest low word that is not zero adds its offset.
        let (one, _) = convert(&access(0x1_0000_0001));
        assert!(one.contains("%5: u32 = IAdd32 %0, u32 0x1"), "{one}");
    }

    /// A structured function converts in place, in its tree's blocks; accesses in two blocks
    /// are collected in post order (block2's first), so its replacement ids come first.
    #[test]
    fn a_structured_function_converts() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x200
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    %4: u1 = INotEqual %3, u32 0x0
block1:
    %5 = WriteGlobal32 %2, u32 0x7
block2:
    %6 = WriteGlobal32 %2, u32 0x9
structure:
    block0
    if %4 {
        block1
    }
    block2
    return
";
        let (out, buffers) = convert(text);
        assert_eq!(
            out,
            "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x200
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %15: u32 = GetCbufU32 u32 0x0, u32 0x200
    %16: u32 = BitwiseAnd32 %15, u32 0xffffffc0
    %17: u32 = ISub32 %0, %16
    %18: u32 = LoadStorage32 u32 0x0, %17
    %3: opaque = Identity %18
    %4: u1 = INotEqual %3, u32 0x0
block1:
    %11: u32 = GetCbufU32 u32 0x0, u32 0x200
    %12: u32 = BitwiseAnd32 %11, u32 0xffffffc0
    %13: u32 = ISub32 %0, %12
    %14 = WriteStorage32 u32 0x0, %13, u32 0x7
    %5 = Void
block2:
    %7: u32 = GetCbufU32 u32 0x0, u32 0x200
    %8: u32 = BitwiseAnd32 %7, u32 0xffffffc0
    %9: u32 = ISub32 %0, %8
    %10 = WriteStorage32 u32 0x0, %9, u32 0x9
    %6 = Void
structure:
    block0
    if %4 {
        block1
    }
    block2
    return
"
        );
        assert_eq!(buffers.descriptors(), [descriptor(0, 0x200, true)]);
    }

    /// The host alignment is the mask's: 1 keeps every bit, 256 clears the low 8.
    #[test]
    fn the_mask_follows_the_alignment() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x100
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    return
";
        for (bytes, mask) in [(1, "0xffffffff"), (256, "0xffffff00")] {
            let mut function = parse(text);
            to_storage_buffers(&mut function, align(bytes)).expect("converts");
            let line = format!("%5: u32 = BitwiseAnd32 %4, u32 {mask}");
            assert!(function.to_string().contains(&line), "{function}");
        }
    }

    /// A function without global memory is left as it is, with an empty table.
    #[test]
    fn a_function_without_global_memory_is_unchanged() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x100
    %1: u32 = LoadStorage32 u32 0x0, %0
    return
";
        let mut function = parse(text);
        let before = function.clone();
        let buffers = to_storage_buffers(&mut function, align(ALIGN)).expect("runs");
        assert_eq!(function, before);
        assert!(buffers.descriptors().is_empty());
    }
}

/// Blocks the entry cannot reach are walked (module docs, departures): one converts and adds
/// its buffer, which Eden, removing the block first, would not; one that does not trace is
/// refused.
#[test]
fn unreachable_blocks_are_walked() {
    let traced = "block0:
    return
block1:
    %0: u32 = GetCbufU32 u32 0x5, u32 0x30
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    return
";
    let (out, buffers) = convert(traced);
    assert!(out.contains("LoadStorage32 u32 0x0"), "{out}");
    assert_eq!(buffers.descriptors(), [descriptor(5, 0x30, false)]);
    let untraced = "block0:
    return
block1:
    %0: u32 = LoadGlobal32 u64 0x1000
    return
";
    assert_eq!(
        refuse(untraced),
        GlobalMemoryError::Untraced {
            at: value(0),
            opcode: Opcode::LoadGlobal32,
            reason: UntracedReason::ImmediateAddress,
        }
    );
}

#[test]
fn alignment_is_a_power_of_two() {
    for bytes in [0, 3, 6, 255, 257, u32::MAX] {
        assert_eq!(
            StorageAlignment::new(bytes),
            Err(AlignmentError(bytes)),
            "{bytes}"
        );
    }
    for shift in 0..u32::BITS {
        let bytes = 1_u32 << shift;
        let alignment = StorageAlignment::new(bytes).expect("a power of two");
        assert_eq!(alignment.bytes(), bytes);
        assert_eq!(alignment.mask(), !(bytes - 1));
    }
    assert_eq!(
        AlignmentError(0).to_string(),
        "a storage buffer alignment of 0 bytes is not a power of two"
    );
}

/// Eden's opcode lists, written out from
/// `eden/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp` @ cf654052
/// (`GlobalToStorage`, with `IsGlobalMemoryWrite` as the last column): every row maps, the
/// written flag is Eden's, and no other opcode maps.
#[test]
fn the_opcode_table_is_edens() {
    const EDEN: [(&str, &str, bool); 50] = [
        ("LoadGlobalS8", "LoadStorageS8", false),
        ("LoadGlobalU8", "LoadStorageU8", false),
        ("LoadGlobalS16", "LoadStorageS16", false),
        ("LoadGlobalU16", "LoadStorageU16", false),
        ("LoadGlobal32", "LoadStorage32", false),
        ("LoadGlobal64", "LoadStorage64", false),
        ("LoadGlobal128", "LoadStorage128", false),
        ("WriteGlobalS8", "WriteStorageS8", true),
        ("WriteGlobalU8", "WriteStorageU8", true),
        ("WriteGlobalS16", "WriteStorageS16", true),
        ("WriteGlobalU16", "WriteStorageU16", true),
        ("WriteGlobal32", "WriteStorage32", true),
        ("WriteGlobal64", "WriteStorage64", true),
        ("WriteGlobal128", "WriteStorage128", true),
        ("GlobalAtomicIAdd32", "StorageAtomicIAdd32", true),
        ("GlobalAtomicSMin32", "StorageAtomicSMin32", true),
        ("GlobalAtomicUMin32", "StorageAtomicUMin32", true),
        ("GlobalAtomicSMax32", "StorageAtomicSMax32", true),
        ("GlobalAtomicUMax32", "StorageAtomicUMax32", true),
        ("GlobalAtomicInc32", "StorageAtomicInc32", true),
        ("GlobalAtomicDec32", "StorageAtomicDec32", true),
        ("GlobalAtomicAnd32", "StorageAtomicAnd32", true),
        ("GlobalAtomicOr32", "StorageAtomicOr32", true),
        ("GlobalAtomicXor32", "StorageAtomicXor32", true),
        ("GlobalAtomicExchange32", "StorageAtomicExchange32", true),
        ("GlobalAtomicIAdd64", "StorageAtomicIAdd64", true),
        ("GlobalAtomicSMin64", "StorageAtomicSMin64", true),
        ("GlobalAtomicUMin64", "StorageAtomicUMin64", true),
        ("GlobalAtomicSMax64", "StorageAtomicSMax64", true),
        ("GlobalAtomicUMax64", "StorageAtomicUMax64", true),
        ("GlobalAtomicAnd64", "StorageAtomicAnd64", true),
        ("GlobalAtomicOr64", "StorageAtomicOr64", true),
        ("GlobalAtomicXor64", "StorageAtomicXor64", true),
        ("GlobalAtomicExchange64", "StorageAtomicExchange64", true),
        ("GlobalAtomicIAdd32x2", "StorageAtomicIAdd32x2", true),
        ("GlobalAtomicSMin32x2", "StorageAtomicSMin32x2", true),
        ("GlobalAtomicUMin32x2", "StorageAtomicUMin32x2", true),
        ("GlobalAtomicSMax32x2", "StorageAtomicSMax32x2", true),
        ("GlobalAtomicUMax32x2", "StorageAtomicUMax32x2", true),
        ("GlobalAtomicAnd32x2", "StorageAtomicAnd32x2", true),
        ("GlobalAtomicOr32x2", "StorageAtomicOr32x2", true),
        ("GlobalAtomicXor32x2", "StorageAtomicXor32x2", true),
        (
            "GlobalAtomicExchange32x2",
            "StorageAtomicExchange32x2",
            true,
        ),
        ("GlobalAtomicAddF32", "StorageAtomicAddF32", true),
        ("GlobalAtomicAddF16x2", "StorageAtomicAddF16x2", true),
        ("GlobalAtomicAddF32x2", "StorageAtomicAddF32x2", true),
        ("GlobalAtomicMinF16x2", "StorageAtomicMinF16x2", true),
        ("GlobalAtomicMinF32x2", "StorageAtomicMinF32x2", true),
        ("GlobalAtomicMaxF16x2", "StorageAtomicMaxF16x2", true),
        ("GlobalAtomicMaxF32x2", "StorageAtomicMaxF32x2", true),
    ];
    for (global, storage, written) in EDEN {
        let global = Opcode::from_name(global).expect("a global opcode");
        let storage = Opcode::from_name(storage).expect("a storage opcode");
        let (mapped, access) = storage_opcode(global).expect("maps");
        assert_eq!(mapped, storage, "{global}");
        assert_eq!(access != Access::Load, written, "{global}");
        // A load keeps its name's kind: an atomic returns, a store does not.
        let expected = match global.name() {
            name if name.starts_with("Load") => Access::Load,
            name if name.starts_with("Write") => Access::Write,
            _ => Access::Atomic,
        };
        assert_eq!(access, expected, "{global}");
        // The storage form returns the same type and takes the binding and offset in place of
        // the address.
        assert_eq!(storage.result(), global.result(), "{global}");
        let tail = global.args().get(1..).unwrap_or_default();
        assert_eq!(
            storage.args().get(..2),
            Some(&[Type::U32, Type::U32][..]),
            "{global}"
        );
        assert_eq!(storage.args().get(2..), Some(tail), "{global}");
    }
    let mapped = Opcode::ALL
        .iter()
        .filter(|&&opcode| storage_opcode(opcode).is_some())
        .count();
    assert_eq!(mapped, EDEN.len());
}

/// The bias and alignments, written out from Eden's `CollectStorageBuffers` (`nvn_bias`) and
/// `Track` (`8U`) and the binding's `Read<u64>`, @ cf654052.
#[test]
fn the_constants_are_edens() {
    assert_eq!(
        NVN_BIAS,
        Bias {
            bank: 0,
            begin: 0x100,
            end: 0x700,
            alignment: 16,
        }
    );
    assert_eq!(UNBIASED_ALIGNMENT, 8);
    assert_eq!(POINTER_BYTES, 8);
}

mod rejections {
    use super::*;

    /// The tail of a function: `%9` is the access, through address `%8`.
    fn through(body: &str) -> String {
        format!("block0:\n{body}    %9: u32 = LoadGlobal32 %8\n    return\n")
    }

    fn untraced(reason: UntracedReason) -> GlobalMemoryError {
        GlobalMemoryError::Untraced {
            at: value(9),
            opcode: Opcode::LoadGlobal32,
            reason,
        }
    }

    #[test]
    fn each_reason_is_refused_exactly() {
        use UntracedReason as R;
        let cases: [(&str, R); 9] = [
            (
                "    %8: opaque = Identity u64 0x1000\n",
                R::ImmediateAddress,
            ),
            (
                "    %0: u32 = GetCbufU32 u32 0x0, u32 0x100\n\
                 \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
                 \x20   %2: u64 = PackUint2x32 %1\n\
                 \x20   %3: u64 = UndefU64\n\
                 \x20   %8: u64 = IAdd64 %2, %3\n",
                R::OffsetNotImmediate,
            ),
            (
                "    %8: u64 = IAdd64 u64 0x1000, u64 0x10\n",
                R::ImmediateBase,
            ),
            ("    %8: u64 = PackUint2x32 u32 0x5\n", R::ImmediatePair),
            ("    %8: u64 = UndefU64\n", R::NotAPair),
            (
                "    %0: u32 = UndefU32\n\
                 \x20   %1: u32x2 = CompositeConstructU32x2 %0, %0\n\
                 \x20   %2: opaque = Identity %1\n\
                 \x20   %3: u32x2 = CompositeInsertU32x2 %2, %0, u32 0x1\n\
                 \x20   %8: u64 = PackUint2x32 %3\n",
                R::NotAPair,
            ),
            (
                "    %0: u32 = GetCbufU32 u32 0x0, u32 0x104\n\
                 \x20   %1: u32x2 = CompositeConstructU32x2 u32 0x1000, %0\n\
                 \x20   %8: u64 = PackUint2x32 %1\n",
                R::ImmediateLowWord,
            ),
            (
                // Bank 0 at 0x104: aligned to neither search's alignment.
                "    %0: u32 = GetCbufU32 u32 0x0, u32 0x104\n\
                 \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
                 \x20   %8: u64 = PackUint2x32 %1\n",
                R::NoPointer,
            ),
            (
                // A register offset: Eden's "TODO: Support SSBO arrays".
                "    %0: u32 = UndefU32\n\
                 \x20   %1: u32 = GetCbufU32 u32 0x0, %0\n\
                 \x20   %2: u32x2 = CompositeConstructU32x2 %1, u32 0x0\n\
                 \x20   %8: u64 = PackUint2x32 %2\n",
                R::NoPointer,
            ),
        ];
        for (body, reason) in cases {
            assert_eq!(refuse(&through(body)), untraced(reason), "{body}");
        }
    }

    /// Each reason's words, and the error's.
    #[test]
    fn the_words_are_plain() {
        use UntracedReason as R;
        let words = [
            (R::ImmediateAddress, "the address is an immediate"),
            (
                R::OffsetNotImmediate,
                "the 64-bit addition's offset is not a u64 immediate",
            ),
            (
                R::ImmediateBase,
                "the 64-bit addition's base is an immediate",
            ),
            (R::ImmediatePair, "the packed pair is an immediate"),
            (
                R::NotAPair,
                "the address is not built from two 32-bit words",
            ),
            (
                R::ImmediateLowWord,
                "the address's low word is an immediate",
            ),
            (
                R::NoPointer,
                "no constant buffer word holds the address's low word",
            ),
        ];
        for (reason, text) in words {
            assert_eq!(reason.to_string(), text);
        }
        assert_eq!(
            untraced(R::NoPointer).to_string(),
            "%9: the address of LoadGlobal32 cannot be traced to a storage buffer pointer: \
             no constant buffer word holds the address's low word"
        );
        assert_eq!(
            GlobalMemoryError::Bank {
                at: value(4),
                opcode: Opcode::WriteGlobal32,
                bank: 32,
            }
            .to_string(),
            "%4: WriteGlobal32 finds its storage buffer pointer in constant buffer bank 32, \
             which does not exist; banks stop at 31"
        );
        assert_eq!(
            GlobalMemoryError::Offset {
                at: value(4),
                opcode: Opcode::LoadGlobal32,
                bank: 2,
                offset: 0xfffc,
            }
            .to_string(),
            "%4: LoadGlobal32 finds its storage buffer pointer at offset 0xfffc of constant \
             buffer bank 2, and its 8 bytes end past the bank's 0x10000"
        );
    }

    /// A load through a pointer at `bank`, `offset`.
    fn at(bank: u32, offset: u32) -> String {
        through(&format!(
            "    %0: u32 = GetCbufU32 u32 {bank:#x}, u32 {offset:#x}\n\
             \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
             \x20   %8: u64 = PackUint2x32 %1\n"
        ))
    }

    #[test]
    fn bank_edges() {
        let last = u32::from(CBUF_BANKS) - 1;
        let (_, buffers) = convert(&at(last, 0x8));
        assert_eq!(
            buffers.descriptors(),
            [descriptor(CBUF_BANKS - 1, 0x8, false)]
        );
        for bank in [last + 1, u32::from(u8::MAX) + 1, u32::MAX] {
            assert_eq!(
                refuse(&at(bank, 0x8)),
                GlobalMemoryError::Bank {
                    at: value(9),
                    opcode: Opcode::LoadGlobal32,
                    bank,
                },
                "{bank}"
            );
        }
    }

    #[test]
    fn offset_edges() {
        let last = INDEXED_CBUF_SIZE - POINTER_BYTES;
        let (_, buffers) = convert(&at(2, last));
        assert_eq!(buffers.descriptors(), [descriptor(2, last, false)]);
        // One alignment step past the last, and an offset whose end does not fit a u32.
        for offset in [last + UNBIASED_ALIGNMENT, u32::MAX - (POINTER_BYTES - 1)] {
            assert_eq!(
                refuse(&at(2, offset)),
                GlobalMemoryError::Offset {
                    at: value(9),
                    opcode: Opcode::LoadGlobal32,
                    bank: 2,
                    offset,
                },
                "{offset:#x}"
            );
        }
    }

    #[test]
    fn bank_is_checked_before_offset() {
        let bank = u32::from(CBUF_BANKS);
        assert_eq!(
            refuse(&at(bank, INDEXED_CBUF_SIZE)),
            GlobalMemoryError::Bank {
                at: value(9),
                opcode: Opcode::LoadGlobal32,
                bank,
            }
        );
    }

    /// Two faulty accesses in a block: the first listed is reported, though its fault (no
    /// pointer) comes later in one access's order than the second's (not a pair). Across
    /// blocks, post order: block1's access before block0's.
    #[test]
    fn the_first_access_in_post_order_is_reported() {
        let one_block = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x104
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    %4: u64 = UndefU64
    %5: u32 = LoadGlobal32 %4
    return
";
        assert_eq!(
            refuse(one_block),
            GlobalMemoryError::Untraced {
                at: value(3),
                opcode: Opcode::LoadGlobal32,
                reason: UntracedReason::NoPointer,
            }
        );
        let two_blocks = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x104
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    jump block1
block1:
    %4: u64 = UndefU64
    %5 = WriteGlobal32 %4, u32 0x1
    return
";
        assert_eq!(
            refuse(two_blocks),
            GlobalMemoryError::Untraced {
                at: value(5),
                opcode: Opcode::WriteGlobal32,
                reason: UntracedReason::NotAPair,
            }
        );
    }

    /// A good access first and a refused one after: nothing is changed (the module docs'
    /// "every refusal happens while collecting"), and the same function converts once the
    /// refused access is gone.
    #[test]
    fn a_refused_function_is_unchanged() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x100
    %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u32 = LoadGlobal32 %2
    %4 = WriteGlobal32 u64 0x40, %3
    return
";
        assert_eq!(
            refuse(text),
            GlobalMemoryError::Untraced {
                at: value(4),
                opcode: Opcode::WriteGlobal32,
                reason: UntracedReason::ImmediateAddress,
            }
        );
        let fixed = text.replace("    %4 = WriteGlobal32 u64 0x40, %3\n", "");
        let (_, buffers) = convert(&fixed);
        assert_eq!(buffers.descriptors(), [descriptor(0, 0x100, false)]);
    }

    /// A bank or offset immediate that is not a `u32` is not a match, and an `IAdd64` offset
    /// that is not a `u64` is not an immediate offset (only a function that does not verify has
    /// either).
    #[test]
    fn wrong_immediate_kinds() {
        let bank = through(
            "    %0: u32 = GetCbufU32 u64 0x0, u32 0x100\n\
             \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
             \x20   %8: u64 = PackUint2x32 %1\n",
        );
        assert_eq!(refuse(&bank), untraced(UntracedReason::NoPointer));
        let offset = through(
            "    %0: u32 = GetCbufU32 u32 0x0, u64 0x100\n\
             \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
             \x20   %8: u64 = PackUint2x32 %1\n",
        );
        assert_eq!(refuse(&offset), untraced(UntracedReason::NoPointer));
        let add = through(
            "    %0: u32 = GetCbufU32 u32 0x0, u32 0x100\n\
             \x20   %1: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
             \x20   %2: u64 = PackUint2x32 %1\n\
             \x20   %8: u64 = IAdd64 %2, u32 0x10\n",
        );
        assert_eq!(refuse(&add), untraced(UntracedReason::OffsetNotImmediate));
    }

    /// An `Identity` chain that loops is a malformed function, refused before anything
    /// changes.
    #[test]
    fn an_identity_cycle_is_refused() {
        let text = through(
            "    %0: opaque = Identity %1\n\
             \x20   %1: opaque = Identity %0\n\
             \x20   %8: u64 = IAdd64 %0, u64 0x10\n",
        );
        assert!(matches!(
            refuse(&text),
            GlobalMemoryError::Pass(PassError::Identity(_))
        ));
    }
}

mod search {
    use std::collections::VecDeque;
    use std::fmt::Write;

    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngSeed};

    use super::*;

    /// The bank and offset a single load in `text` (through low word `%9`) finds.
    fn found(body: &str) -> (u8, u32) {
        let text = format!(
            "block0:\n{body}    %10: u32x2 = CompositeConstructU32x2 %9, u32 0x0\n\
             \x20   %11: u64 = PackUint2x32 %10\n\
             \x20   %12: u32 = LoadGlobal32 %11\n    return\n"
        );
        let (_, buffers) = convert(&text);
        let [only] = buffers.descriptors() else {
            panic!("one buffer: {:?}", buffers.descriptors());
        };
        (only.bank(), only.offset())
    }

    #[test]
    fn the_bias_is_tried_first_even_when_farther() {
        // Unbiased bank 3 at depth 1, biased bank 0 at 0x100 at depth 2.
        let body = "    %0: u32 = GetCbufU32 u32 0x3, u32 0x8
    %1: u32 = GetCbufU32 u32 0x0, u32 0x100
    %2: u32 = IAdd32 %1, u32 0x0
    %9: u32 = IAdd32 %2, %0
";
        assert_eq!(found(body), (0, 0x100));
    }

    #[test]
    fn the_nearest_match_wins_over_the_rightmost() {
        // Bank 2 at 0x10 on the left at depth 1; bank 2 at 0x18 under the right at depth 2.
        let body = "    %0: u32 = GetCbufU32 u32 0x2, u32 0x10
    %1: u32 = GetCbufU32 u32 0x2, u32 0x18
    %2: u32 = IAdd32 %1, u32 0x1
    %9: u32 = IAdd32 %0, %2
";
        assert_eq!(found(body), (2, 0x10));
    }

    #[test]
    fn among_the_nearest_the_rightmost_wins() {
        let body = "    %0: u32 = GetCbufU32 u32 0x2, u32 0x10
    %1: u32 = GetCbufU32 u32 0x2, u32 0x18
    %9: u32 = IAdd32 %0, %1
";
        assert_eq!(found(body), (2, 0x18));
        // Two levels down, Eden's queue holds the right operand's arguments (right to left)
        // before the left's: bank 4's two, then bank 5's.
        let body = "    %0: u32 = GetCbufU32 u32 0x5, u32 0x10
    %1: u32 = GetCbufU32 u32 0x5, u32 0x18
    %2: u32 = GetCbufU32 u32 0x4, u32 0x20
    %3: u32 = GetCbufU32 u32 0x4, u32 0x28
    %4: u32 = IAdd32 %0, %1
    %5: u32 = IAdd32 %2, %3
    %9: u32 = IAdd32 %4, %5
";
        assert_eq!(found(body), (4, 0x28));
    }

    #[test]
    fn a_phi_cycle_is_searched() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x120
    jump block1
block1:
    %1: u32 = Phi [block0: %0], [block1: %2]
    %2: u32 = IAdd32 %1, u32 0x10
    %3: u32x2 = CompositeConstructU32x2 %2, u32 0x0
    %4: u64 = PackUint2x32 %3
    %5 = WriteGlobal32 %4, %2
    %6: u1 = INotEqual %2, u32 0x0
    branch %6, block1, block2
block2:
    return
";
        let (_, buffers) = convert(text);
        assert_eq!(buffers.descriptors(), [descriptor(0, 0x120, true)]);
    }

    /// Every edge of each search's acceptance, Eden's `pred` with and without the bias.
    #[test]
    fn acceptance_edges() {
        let biased = [
            (0, 0x0f0, false),
            (0, 0x100, true),
            (0, 0x108, false),
            (0, 0x6f0, true),
            (0, 0x700, false),
            (1, 0x100, false),
            (u32::MAX, 0x100, false),
        ];
        for (bank, offset, accepted) in biased {
            assert_eq!(
                Search::Biased.accepts(bank, offset),
                accepted,
                "{bank} {offset:#x}"
            );
        }
        let unbiased = [
            (0, 0x0, true),
            (7, 0x4, false),
            (7, 0x8, true),
            (7, 0xc, false),
            (7, 0x108, true),
            (u32::MAX, 0xffff_fff8, true),
        ];
        for (bank, offset, accepted) in unbiased {
            assert_eq!(
                Search::Unbiased.accepts(bank, offset),
                accepted,
                "{bank} {offset:#x}"
            );
        }
    }

    /// A direct port of Eden's `BreadthFirstSearch` (`frontend/ir/breadth_first_search.h`
    /// @ cf654052) with `Track`'s predicate: the queue, the visited list (the start is not in
    /// it), arguments right to left, `Identity` looked through.
    fn eden_search(function: &Function, start: Value, search: Search) -> Option<u32> {
        let resolve = |operand: Operand| -> Operand {
            let mut current = operand;
            loop {
                match current {
                    Operand::Value(v) => match function.inst(v) {
                        Some(Inst::Op(op)) if op.opcode == Opcode::Identity => {
                            current = *op.args.first().expect("an argument");
                        }
                        _ => return current,
                    },
                    Operand::Imm(_) => return current,
                }
            }
        };
        let Operand::Value(first) = resolve(Operand::Value(start)) else {
            return None;
        };
        let mut visited: Vec<Value> = Vec::new();
        let mut queue = VecDeque::from([first]);
        while let Some(inst) = queue.pop_front() {
            let args: Vec<Operand> = match function.inst(inst).expect("exists") {
                Inst::Op(op) => {
                    if matches!(op.opcode, Opcode::GetCbufU32 | Opcode::GetCbufU32x2)
                        && let Some(Operand::Imm(Imm::U32(bank))) =
                            op.args.first().map(|&a| resolve(a))
                        && let Some(Operand::Imm(Imm::U32(offset))) =
                            op.args.get(1).map(|&a| resolve(a))
                        && search.accepts(bank, offset)
                    {
                        return Some(inst.index());
                    }
                    op.args.clone()
                }
                Inst::Phi(phi) => phi.incoming.iter().map(|&(_, a)| a).collect(),
            };
            for &arg in args.iter().rev() {
                if let Operand::Value(next) = resolve(arg)
                    && !visited.contains(&next)
                {
                    visited.push(next);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// One generated instruction.
    #[derive(Clone, Debug)]
    enum Gen {
        /// A constant-buffer read of a bank and an offset from small sets.
        Cbuf(u32, u32),
        /// An add of two earlier values (by distance back, wrapped to what exists).
        Add(usize, usize),
        /// An add of an earlier value and an immediate.
        AddImm(usize),
        /// An `Identity` of an earlier value.
        Identity(usize),
        /// A select of two earlier values.
        Select(usize, usize),
        /// A value that leads nowhere.
        Undef,
    }

    fn any_gen() -> impl Strategy<Value = Gen> {
        prop_oneof![
            (
                0_u32..3,
                prop::sample::select(vec![0x4_u32, 0x8, 0x100, 0x108, 0x110, 0x6f0, 0x700])
            )
                .prop_map(|(bank, offset)| Gen::Cbuf(bank, offset)),
            (1_usize..6, 1_usize..6).prop_map(|(a, b)| Gen::Add(a, b)),
            (1_usize..6).prop_map(Gen::AddImm),
            (1_usize..4).prop_map(Gen::Identity),
            (1_usize..6, 1_usize..6).prop_map(|(a, b)| Gen::Select(a, b)),
            Just(Gen::Undef),
        ]
    }

    /// A function of the generated instructions: `head` in block0, then a loop block whose
    /// `phis` take one value from block0 and one from the loop's own `body` (cycles).
    fn function(head: &[Gen], phis: &[(usize, usize)], body: &[Gen]) -> Function {
        let mut text = String::from("block0:\n    %0: u1 = UndefU1\n    %1: u32 = UndefU32\n");
        let mut id = 2_usize;
        let write = |text: &mut String, id: &mut usize, item: &Gen| {
            let back = |d: usize| (*id).saturating_sub(d).max(1);
            let line = match *item {
                Gen::Cbuf(bank, offset) => {
                    format!("u32 = GetCbufU32 u32 {bank:#x}, u32 {offset:#x}")
                }
                Gen::Add(a, b) => format!("u32 = IAdd32 %{}, %{}", back(a), back(b)),
                Gen::AddImm(a) => format!("u32 = IAdd32 %{}, u32 0x4", back(a)),
                Gen::Identity(a) => format!("opaque = Identity %{}", back(a)),
                Gen::Select(a, b) => format!("u32 = SelectU32 %0, %{}, %{}", back(a), back(b)),
                Gen::Undef => "u32 = UndefU32".to_string(),
            };
            let _ = writeln!(text, "    %{id}: {line}");
            *id += 1;
        };
        for item in head {
            write(&mut text, &mut id, item);
        }
        text.push_str("    jump block1\nblock1:\n");
        let head_end = id;
        let first_phi = id;
        let body_start = first_phi + phis.len();
        let body_len = body.len().max(1);
        for &(from_head, from_body) in phis {
            let incoming = (head_end - 1 - from_head % (head_end - 1)).max(1);
            let looped = body_start + from_body % body_len;
            let _ = writeln!(
                text,
                "    %{id}: u32 = Phi [block0: %{incoming}], [block1: %{looped}]"
            );
            id += 1;
        }
        if body.is_empty() {
            let _ = writeln!(text, "    %{id}: u32 = UndefU32");
            id += 1;
        }
        for item in body {
            write(&mut text, &mut id, item);
        }
        text.push_str("    branch %0, block1, block2\nblock2:\n    return\n");
        Function::parse(&text).expect("parses")
    }

    fn config() -> Config {
        Config {
            cases: 256,
            rng_seed: RngSeed::Fixed(0x5107_0014),
            failure_persistence: None,
            ..Config::default()
        }
    }

    proptest! {
        #![proptest_config(config())]

        /// The search finds what Eden's breadth-first search finds from every instruction,
        /// whether the tracker was built from all starts or from that start alone.
        #[test]
        fn the_search_matches_edens_breadth_first_search(
            head in prop::collection::vec(any_gen(), 1..24),
            phis in prop::collection::vec((0_usize..32, 0_usize..32), 0..4),
            body in prop::collection::vec(any_gen(), 0..16),
        ) {
            let function = function(&head, &phis, &body);
            let starts: Vec<Value> = function
                .listed()
                .filter(|(_, inst)| inst.result() != Type::Void)
                .map(|(v, _)| v)
                .collect();
            let mut work = Work::default();
            let mut resolver = Resolver::new(&function);
            let mut all = Tracker::new(&function, &mut resolver, &starts, &mut work)
                .expect("builds");
            for &start in &starts {
                for search in [Search::Biased, Search::Unbiased] {
                    let eden = eden_search(&function, start, search);
                    prop_assert_eq!(all.find(start, search, &mut work), eden, "{} {:?}\n{}", start, search, function);
                    let mut alone = Tracker::new(&function, &mut resolver, &[start], &mut work)
                        .expect("builds");
                    prop_assert_eq!(alone.find(start, search, &mut work), eden, "{} {:?} alone", start, search);
                }
            }
        }
    }
}

mod equivalence {
    use super::*;

    /// Where the test's global memory starts.
    const BASE: u64 = 0x2_0000_0000;
    /// Each buffer's window, in bytes: the storage buffer a test gives is this much of global
    /// memory from the pointer aligned down.
    const WINDOW: usize = 0x100;

    /// A machine with global memory of distinct bytes and `pointers` (bank, offset, address)
    /// in their constant buffers.
    fn machine(pointers: &[(u32, u32, u64)]) -> Machine {
        let mut machine = Machine::new();
        let bytes = (0..0x1000_u32)
            .map(|k| u8::try_from(k.wrapping_mul(37) % 251).expect("small"))
            .collect();
        machine.global = Region { base: BASE, bytes };
        for &(bank, offset, address) in pointers {
            let cbuf = machine.cbufs.entry(bank).or_default();
            let end = usize::try_from(offset).expect("small") + 8;
            if cbuf.len() < end {
                cbuf.resize(end, 0);
            }
            let at = usize::try_from(offset).expect("small");
            cbuf.get_mut(at..end)
                .expect("sized")
                .copy_from_slice(&address.to_le_bytes());
        }
        machine
    }

    /// The window of global memory a buffer at `address` covers.
    fn window(machine: &Machine, address: u64, alignment: u32) -> Vec<u8> {
        let start = address & !u64::from(alignment - 1);
        let from = usize::try_from(start - machine.global.base).expect("small");
        machine.global.bytes[from..from + WINDOW].to_vec()
    }

    /// Runs `before` with global memory and its conversion with storage buffers holding the
    /// same windows, and checks the outputs and every window agree after the run.
    fn check(text: &str, pointers: &[(u32, u32, u64)], alignment: u32) {
        let before = parse(text);
        let mut after = before.clone();
        let buffers = to_storage_buffers(&mut after, align(alignment)).expect("converts");
        let mut global = machine(pointers);
        let mut storage = global.clone();
        for (binding, d) in (0_u32..).zip(buffers.descriptors()) {
            let &(_, _, address) = pointers
                .iter()
                .find(|&&(bank, offset, _)| bank == u32::from(d.bank()) && offset == d.offset())
                .expect("a pointer for each buffer");
            storage
                .storage
                .insert(binding, window(&global, address, alignment));
        }
        global.run(&before).expect("runs with global memory");
        storage.run(&after).expect("runs with storage buffers");
        assert_eq!(storage.outputs, global.outputs);
        for (binding, d) in (0_u32..).zip(buffers.descriptors()) {
            let &(_, _, address) = pointers
                .iter()
                .find(|&&(bank, offset, _)| bank == u32::from(d.bank()) && offset == d.offset())
                .expect("a pointer");
            assert_eq!(
                storage.storage.get(&binding),
                Some(&window(&global, address, alignment)),
                "binding {binding}"
            );
        }
    }

    /// [`MIXED`] with pointers that are not aligned to the host alignment (bank 3's is 8 past
    /// a 64-byte boundary), so the mask and the immediate offsets both matter.
    #[test]
    fn mixed_accesses_keep_the_interpreters_result() {
        let pointers = [
            (0, 0x110, BASE + 0x100),
            (3, 0x8, BASE + 0x208),
            (1, 0x40, BASE + 0x340),
        ];
        check(MIXED, &pointers, ALIGN);
        check(MIXED, &pointers, 16);
        check(MIXED, &pointers, 1);
    }

    /// A pointer that walks a loop: each iteration stores through the pointer plus 4 times the
    /// count, read back with a 64-bit load.
    #[test]
    fn a_loop_keeps_the_interpreters_result() {
        let text = "block0:
    %0: u32 = GetCbufU32 u32 0x0, u32 0x180
    %1: u32 = GetCbufU32 u32 0x0, u32 0x184
    jump block1
block1:
    %2: u32 = Phi [block0: %0], [block1: %3]
    %3: u32 = IAdd32 %2, u32 0x4
    %4: u32x2 = CompositeConstructU32x2 %3, %1
    %5: u64 = PackUint2x32 %4
    %6 = WriteGlobal32 %5, %3
    %7: u32 = ISub32 %3, %0
    %8: u1 = ULessThan %7, u32 0x20
    branch %8, block1, block2
block2:
    %9: u32x2 = CompositeConstructU32x2 %0, %1
    %10: u64 = PackUint2x32 %9
    %11: u64 = IAdd64 %10, u64 0x8
    %12: u32x2 = LoadGlobal64 %11
    %13: u32 = CompositeExtractU32x2 %12, u32 0x1
    %14: f32 = BitCastF32U32 %13
    %15 = SetAttribute a0x80, %14, u32 0x0
    return
";
        check(text, &[(0, 0x180, BASE + 0x480)], ALIGN);
    }

    /// Instructions from 12e's translator: `LDC.64 R2, c[0][0x110]`, `LDG.E R4,
    /// [R2.64+0x10]`, `STG.E [R2.64+0x20], R4`, `RED.E.ADD [R2.64+0x24], R4`, translated in one
    /// block (the translation driver does not dispatch the memory group yet), then the slice 8
    /// passes and this pass: the storage accesses do what the global ones did.
    #[test]
    fn a_translated_program_keeps_the_interpreters_result() {
        use crate::decode::{
            AtomDst, AtomOp, AtomType, CBuf, CBufRef, GlobalAccess, Gpr, Instr, LdcMode, MemAddr,
            MemEvictionPriority, MemOrder, MemSpace, MemType, Op, OpAtom, OpLd, OpLdc, OpSt, Pred,
            PredSrc, UniformAddr, UniformReg,
        };
        use crate::ir::{Builder, Terminator};
        use crate::translate::integer::GuardMode;
        use crate::translate::memory::translate_memory;

        const GLOBAL: MemSpace = MemSpace::Global(GlobalAccess {
            order: MemOrder::Weak,
            eviction: MemEvictionPriority::Normal,
        });
        let pair = |offset: i32| MemAddr {
            reg: Gpr::new(2),
            reg_64: true,
            uniform: None,
            offset,
        };
        let program = [
            Op::Ldc(OpLdc {
                dst: Gpr::new(2),
                cbuf: CBufRef {
                    buf: CBuf::Binding(0),
                    offset: 0x110,
                },
                offset: Gpr::RZ,
                mode: LdcMode::Indexed,
                mem_type: MemType::B64,
            }),
            Op::Ld(OpLd {
                dst: Gpr::new(4),
                addr: pair(0x10),
                mem_type: MemType::B32,
                space: GLOBAL,
                pred: PredSrc::TRUE,
                pred_dst: Pred::PT,
                zd: false,
                bit76: false,
            }),
            Op::St(OpSt {
                data: Gpr::new(4),
                addr: pair(0x20),
                mem_type: MemType::B32,
                space: GLOBAL,
                bit76: false,
            }),
            // `RED` reads its register pair from bit 72, which the decoder carries as the
            // uniform register's `.E` (12e's module docs).
            Op::Atom(OpAtom {
                dst: AtomDst::Reduction {
                    ignored: Gpr::new(0),
                },
                addr: MemAddr {
                    uniform: Some(UniformAddr {
                        reg: UniformReg::new(0).expect("a register"),
                        wide: true,
                        enabled: false,
                    }),
                    reg_64: false,
                    ..pair(0x24)
                },
                data: Gpr::new(4),
                op: AtomOp::Add,
                atom_type: AtomType::U32,
                space: GLOBAL,
                pred_dst: Pred::PT,
            }),
        ];
        let mut before = Function::new();
        let entry = before.entry();
        let mut builder = Builder::new(&mut before, entry);
        for op in program {
            translate_memory(&mut builder, &Instr::new(op), GuardMode::Honour).expect("translates");
        }
        builder.terminate(Terminator::Return).expect("terminates");
        crate::opt::optimize(&mut before).expect("optimizes");
        let mut after = before.clone();
        let buffers = to_storage_buffers(&mut after, align(ALIGN)).expect("converts");
        assert_eq!(buffers.descriptors(), [descriptor(0, 0x110, true)]);
        let text = after.to_string();
        assert!(!text.contains("Global"), "{text}");

        let address = BASE + 0x228;
        let mut global = machine(&[(0, 0x110, address)]);
        let mut storage = global.clone();
        storage.storage.insert(0, window(&global, address, ALIGN));
        global.run(&before).expect("runs with global memory");
        storage.run(&after).expect("runs with storage buffers");
        assert_eq!(storage.regs, global.regs);
        assert_eq!(
            storage.storage.get(&0),
            Some(&window(&global, address, ALIGN))
        );
        // The run changed memory: the store and the reduction reached it.
        assert_ne!(global.global.bytes, machine(&[]).global.bytes);
    }
}

mod linear {
    use std::fmt::Write;

    use super::*;
    use crate::translate::driver::MAX_TRANSLATED_INSTRUCTIONS;

    /// Instructions per access in [`chain`]: the pointer step, the pair, the pack and the load.
    const PER_ACCESS: usize = 4;
    /// The instructions before the first access.
    const HEAD: usize = 3;

    /// Row 121's shape: `count` accesses through a pointer that each step moves by a value,
    /// so every access's low word reaches the pointer's read only through every step before it
    /// (Eden's search walks the whole chain again for each access).
    fn chain(count: usize) -> Function {
        let mut text = String::from(
            "block0:\n    %0: u32 = GetCbufU32 u32 0x0, u32 0x100\n    %1: u32 = UndefU32\n    %2: u32 = UndefU32\n",
        );
        let mut previous = 0;
        let mut id = HEAD;
        for _ in 0..count {
            let step = id;
            let _ = writeln!(text, "    %{step}: u32 = IAdd32 %{previous}, %1");
            let _ = writeln!(
                text,
                "    %{}: u32x2 = CompositeConstructU32x2 %{step}, %2",
                step + 1
            );
            let _ = writeln!(text, "    %{}: u64 = PackUint2x32 %{}", step + 2, step + 1);
            let _ = writeln!(text, "    %{}: u32 = LoadGlobal32 %{}", step + 3, step + 2);
            previous = step;
            id += PER_ACCESS;
        }
        text.push_str("    return\n");
        Function::parse(&text).expect("parses")
    }

    /// The pass's work on [`chain`] of `count` accesses.
    fn work(count: usize) -> u64 {
        let mut function = chain(count);
        let buffers = to_storage_buffers(&mut function, align(ALIGN)).expect("converts");
        assert_eq!(buffers.descriptors(), [descriptor(0, 0x100, false)]);
        buffers.work().steps()
    }

    /// The steps per instruction of the input the pass may take: each instruction and argument
    /// is visited a fixed number of times (module docs, "Cost"); the replacements add five
    /// instructions per access.
    const STEPS_PER_INSTRUCTION: u64 = 16;

    #[test]
    fn the_work_at_the_drivers_cap_is_linear() {
        let cap = usize::try_from(MAX_TRANSLATED_INSTRUCTIONS).expect("fits");
        let count = (cap - HEAD) / PER_ACCESS;
        let size = HEAD + count * PER_ACCESS;
        assert!(size <= cap && size + PER_ACCESS > cap, "at the cap");
        let steps = work(count);
        let bound = STEPS_PER_INSTRUCTION * u64::try_from(size).expect("fits");
        assert!(steps <= bound, "{steps} steps for {size} instructions");
        // Eden's search costs about the square of the accesses here.
        let quadratic = u64::try_from(count * count / 2).expect("fits");
        assert!(steps * 100 < quadratic, "{steps} against {quadratic}");
    }

    #[test]
    fn twice_the_accesses_cost_twice_the_work() {
        let small = work(2_000);
        let large = work(4_000);
        assert!(
            large <= 2 * small + 64,
            "{large} for twice, {small} for once"
        );
        assert!(
            large >= 2 * small - 64,
            "{large} for twice, {small} for once"
        );
    }
}
