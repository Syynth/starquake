//! Resources (the `resources` submodule): the binding convention and resource table, the
//! constant-buffer reads against the interpreter on the host's GPU, and each check.

use super::ops::{ONE_INVOCATION, op, u};
use super::run::{gpu_registers_with, opaque};
use std::fmt::Write as _;

use super::*;
use crate::host::{ResourceKind, ResourceSlot, ResourceSource};
use crate::info::{CBUF_ALIGNMENT, INDEXED_CBUF_SIZE};
use crate::interp::Machine;
use crate::ir::Builder;

/// `inputs` with bank `bank` given `size` bytes.
fn with_bank(inputs: LowerInputs, bank: u8, size: u32) -> LowerInputs {
    let mut constant_buffers = inputs.constant_buffers;
    constant_buffers[usize::from(bank)] = size;
    LowerInputs {
        constant_buffers,
        ..inputs
    }
}

/// A function of `stage` that reads a word of each bank of `banks` at offset 0 and writes the
/// last one read to colour target 0 (pixel), generic 0 (vertex) or register 0 (compute).
fn reading(stage: GuestStage, banks: &[u8]) -> Function {
    let mut text = String::from("block0:\n");
    for (index, bank) in banks.iter().enumerate() {
        writeln!(text, "    %{index}: f32 = GetCbufF32 u32 {bank}, u32 0")
            .expect("writes to a string");
    }
    let last = banks.len().checked_sub(1).expect("a bank");
    let generic_0 = u32::from(GENERIC_0_X) * ATTRIBUTE_WORD_BYTES;
    let next = banks.len();
    match stage {
        GuestStage::Pixel => writeln!(text, "    %{next} = SetFragColor u32 0, u32 0, %{last}"),
        GuestStage::Vertex => {
            writeln!(
                text,
                "    %{next} = SetAttribute a{generic_0:#x}, %{last}, u32 0"
            )
        }
        _ => writeln!(
            text,
            "    %{next}: u32 = BitCastU32F32 %{last}\n    %{} = SetRegister r0, %{next}",
            next + 1
        ),
    }
    .expect("writes to a string");
    text.push_str("    return\n");
    ir(&text)
}

/// The resource table of a function reading banks 0, 3 and 31 in each stage: one uniform
/// buffer per bank, ascending, in the stage's group at binding `1 + bank`, with the size the
/// inputs give; a bank the function does not read (bank 7, given a size) is not listed. The
/// module declares exactly the table's bindings, each a uniform array of `size / 16` vectors.
/// The literals are the convention's, written out (module docs, "The binding convention").
#[test]
fn resource_tables_are_golden() {
    let mut inputs = with_bank(ONE_INVOCATION, 0, 0x30);
    inputs = with_bank(inputs, 3, 0x10);
    inputs = with_bank(inputs, 31, INDEXED_CBUF_SIZE);
    inputs = with_bank(inputs, 7, 0x20);
    for (stage, group) in [
        (GuestStage::Vertex, 0),
        (GuestStage::Pixel, 1),
        (GuestStage::Compute, 0),
    ] {
        let shader = lowered(stage, &reading(stage, &[31, 0, 3]), &inputs);
        let slot = |binding, size, bank| ResourceSlot {
            group,
            binding,
            kind: ResourceKind::UniformBuffer { size },
            source: ResourceSource::ConstantBuffer { bank },
        };
        assert_eq!(
            shader.resources,
            [slot(1, 0x30, 0), slot(4, 0x10, 3), slot(32, 0x1_0000, 31)],
            "{stage}"
        );
        assert_eq!(shader.driver_constants, DriverConstantsLayout::CURRENT);
        let module = naga_module(&shader);
        let declared: Vec<(u32, u32, Option<u32>)> = module
            .global_variables
            .iter()
            .filter_map(|(_, global)| {
                let binding = global.binding?;
                let length = match module.types[global.ty].inner {
                    TypeInner::Array {
                        size: naga::ArraySize::Constant(length),
                        stride: 16,
                        ..
                    } => Some(length.get()),
                    _ => None,
                };
                (global.space == naga::AddressSpace::Uniform).then_some((
                    binding.group,
                    binding.binding,
                    length,
                ))
            })
            .collect();
        assert_eq!(
            declared,
            [
                (group, 1, Some(3)),
                (group, 4, Some(1)),
                (group, 32, Some(0x1000))
            ],
            "{stage}: declared in binding order"
        );
    }
}

/// The convention's functions against the literals of the module docs' table, and the size
/// table from collect info.
#[test]
fn the_binding_convention_is_the_documented_one() {
    assert_eq!(resource_group(HostStage::Vertex), 0);
    assert_eq!(resource_group(HostStage::Fragment), 1);
    assert_eq!(resource_group(HostStage::Compute), 0);
    assert_eq!(DRIVER_CONSTANTS_BINDING, 0);
    assert_eq!(constant_buffer_binding(0), 1);
    assert_eq!(constant_buffer_binding(1), 2);
    assert_eq!(constant_buffer_binding(31), 32);
    assert_eq!(CBUF_BANKS, 32);
    assert_eq!(CONSTANT_BUFFER_SLOTS, 32);
    // Collect info's sizes, by bank: a read at 0x24 of bank 2 is rounded up to 0x30, an
    // indexed read of bank 9 is the whole bank.
    let function = ir("block0:\n\
         \x20   %0: u32 = GetCbufU32 u32 2, u32 0x24\n\
         \x20   %1: u32 = GetCbufU32 u32 9, %0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   return\n");
    let info = crate::info::collect(GuestStage::Compute, None, &function).expect("collects");
    let mut expected = [0; CONSTANT_BUFFER_SLOTS];
    expected[2] = 0x30;
    expected[9] = INDEXED_CBUF_SIZE;
    assert_eq!(constant_buffer_sizes(&info), expected);
}

/// The bytes of a bank: byte `i` is `0x80 | (i * 7)`, so every byte has its sign bit set and
/// bytes of one word differ above the low byte.
fn bank_bytes(length: u32) -> Vec<u8> {
    (0..length)
        .map(|i| 0x80 | u8::try_from((i * 7) % 0x80).expect("below 0x80"))
        .collect()
}

/// The bank the GPU comparison reads and its declared size (three elements).
const BANK: u8 = 5;
const SIZE: u32 = 3 * 16;

/// Every read width at offsets inside the declared size: aligned, unaligned (aligned down), at
/// each byte and half-word of a word, and the last bytes; then at and past the size, where the
/// buffer bound has bytes (64 more) that a missing bound would read. Each register holds one
/// case's result; a pair's two words go to two registers.
fn read_cases() -> Vec<(String, Opcode, u32)> {
    let opcodes = [
        Opcode::GetCbufU8,
        Opcode::GetCbufS8,
        Opcode::GetCbufU16,
        Opcode::GetCbufS16,
        Opcode::GetCbufU32,
        Opcode::GetCbufF32,
        Opcode::GetCbufU32x2,
    ];
    let offsets = [
        0,
        1,
        2,
        3,
        6,
        7,
        13,
        20,
        SIZE - 8,
        SIZE - 5,
        SIZE - 1,
        SIZE,
        SIZE + 1,
        SIZE + 8,
        u32::MAX,
    ];
    opcodes
        .iter()
        .flat_map(|&opcode| {
            offsets
                .iter()
                .map(move |&offset| (format!("{opcode} at {offset:#x}"), opcode, offset))
        })
        .collect()
}

/// A compute function that writes each case of `cases` to its registers, and the registers.
fn read_function(cases: &[(String, Opcode, u32)]) -> (Function, Vec<u8>) {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let mut regs = Vec::new();
    let mut next = 0_u8;
    let mut set = |builder: &mut Builder<'_>, value: Value, regs: &mut Vec<u8>| {
        op(
            builder,
            Opcode::SetRegister,
            &[Operand::Imm(Imm::Reg(Gpr::new(next))), value.into()],
        );
        regs.push(next);
        next += 1;
    };
    for (_, opcode, offset) in cases {
        let read = op(&mut builder, *opcode, &[u(u32::from(BANK)), u(*offset)]);
        match opcode {
            Opcode::GetCbufF32 => {
                let bits = op(&mut builder, Opcode::BitCastU32F32, &[read.into()]);
                set(&mut builder, bits, &mut regs);
            }
            Opcode::GetCbufU32x2 => {
                for component in 0..2 {
                    let word = op(
                        &mut builder,
                        Opcode::CompositeExtractU32x2,
                        &[read.into(), u(component)],
                    );
                    set(&mut builder, word, &mut regs);
                }
            }
            _ => set(&mut builder, read, &mut regs),
        }
    }
    builder
        .terminate(Terminator::Return)
        .expect("the block returns");
    (function, regs)
}

/// The words a case gives: the interpreter's over the declared bytes for an offset inside the
/// size, and zero (one word, or two for a pair) for one at or past it.
fn expected_words(function: &Function, regs: &[u8], cases: &[(String, Opcode, u32)]) -> Vec<u32> {
    // The interpreter reads each case alone, so a case past the size, which it refuses, does
    // not stop the others.
    let mut words = Vec::new();
    for (name, opcode, offset) in cases {
        let count = if *opcode == Opcode::GetCbufU32x2 {
            2
        } else {
            1
        };
        if *offset >= SIZE {
            words.extend(std::iter::repeat_n(0, count));
            continue;
        }
        let (single, single_regs) = read_function(&[(name.clone(), *opcode, *offset)]);
        let mut machine = Machine::new();
        machine.cbufs.insert(u32::from(BANK), bank_bytes(SIZE));
        machine
            .run(&single)
            .expect("the interpreter reads inside the bank");
        words.extend(
            single_regs
                .iter()
                .map(|&reg| machine.regs[usize::from(reg)]),
        );
    }
    assert_eq!(words.len(), regs.len(), "{}", function.insts().count());
    words
}

/// Each read width at each offset of [`read_cases`], with offsets the host compiler cannot
/// fold, gives on the host's GPU what the interpreter gives inside the declared size, and zero
/// at and past it although the bound buffer has bytes there (module docs, "Bounds").
#[test]
fn reads_match_the_interpreter_on_the_gpu() {
    let cases = read_cases();
    let (function, regs) = read_function(&cases);
    let expected = expected_words(&function, &regs, &cases);
    let function = opaque(&function);
    let shader = lowered(
        GuestStage::Compute,
        &function,
        &with_bank(ONE_INVOCATION, BANK, SIZE),
    );
    let bytes = bank_bytes(SIZE + 64);
    let Some(found) = gpu_registers_with(
        "reads_match_the_interpreter_on_the_gpu",
        &shader,
        &regs,
        &[(constant_buffer_binding(BANK), &bytes)],
    ) else {
        return;
    };
    let mismatches: Vec<String> = regs
        .iter()
        .zip(expected.iter().zip(&found))
        .filter(|(_, (expected, found))| expected != found)
        .map(|(reg, (expected, found))| format!("r{reg}: expected {expected:#x}, GPU {found:#x}"))
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The reads' MSL, one module per width with an immediate offset and a register offset each:
/// the alignment mask, the bound against the declared size, the element and word indices and,
/// for the narrow reads, the field (`tests::resources::reads_match_the_interpreter_on_the_gpu`
/// runs the same arithmetic on the GPU; this pins it where no GPU is).
#[test]
fn reads_snapshot() {
    let function = ir("block0:\n\
         \x20   %0: u32 = GetCbufU32 u32 5, u32 0x24\n\
         \x20   %1: u32 = GetCbufS8 u32 5, %0\n\
         \x20   %2: u32 = GetCbufU16 u32 5, %0\n\
         \x20   %3: f32 = GetCbufF32 u32 5, u32 0x2c\n\
         \x20   %4: u32x2 = GetCbufU32x2 u32 5, %0\n\
         \x20   %5: u32 = CompositeExtractU32x2 %4, u32 1\n\
         \x20   %6: u32 = BitCastU32F32 %3\n\
         \x20   %7: u32 = IAdd32 %1, %2\n\
         \x20   %8: u32 = IAdd32 %5, %6\n\
         \x20   %9: u32 = IAdd32 %7, %8\n\
         \x20   %10 = SetRegister r0, %9\n\
         \x20   return\n");
    let shader = lowered(
        GuestStage::Compute,
        &function,
        &with_bank(ONE_INVOCATION, 5, SIZE),
    );
    insta::assert_snapshot!("constant_buffer_reads", msl(&shader));
}

/// A bank whose size is not a multiple of the alignment or is past the whole bank is refused
/// before anything else, the lowest bank first, at each edge.
#[test]
fn bank_sizes_are_checked_at_their_edges() {
    let function = reading(GuestStage::Compute, &[0]);
    let lowers = |inputs: &LowerInputs| lower_with(GuestStage::Compute, &function, inputs).err();
    for size in [CBUF_ALIGNMENT, INDEXED_CBUF_SIZE] {
        assert_eq!(lowers(&with_bank(ONE_INVOCATION, 0, size)), None, "{size}");
    }
    for size in [
        CBUF_ALIGNMENT - 1,
        CBUF_ALIGNMENT + 1,
        INDEXED_CBUF_SIZE + CBUF_ALIGNMENT,
        INDEXED_CBUF_SIZE - 1,
    ] {
        assert_eq!(
            lowers(&with_bank(with_bank(ONE_INVOCATION, 0, 16), 4, size)),
            Some(LowerError::ConstantBufferSize { bank: 4, size }),
            "{size}"
        );
    }
    // Two faulty banks: the lowest is reported. A faulty bank the function does not read is
    // still refused.
    let both = with_bank(with_bank(with_bank(ONE_INVOCATION, 0, 16), 9, 17), 30, 3);
    assert_eq!(
        lowers(&both),
        Some(LowerError::ConstantBufferSize { bank: 9, size: 17 })
    );
    // The size check (step 3) comes before the read's own checks (step 5): a register bank
    // with a faulty size elsewhere reports the size.
    let register_bank = ir("block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetCbufU32 %0, u32 0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   return\n");
    assert_eq!(
        lower_with(
            GuestStage::Compute,
            &register_bank,
            &with_bank(ONE_INVOCATION, 2, 8)
        )
        .err(),
        Some(LowerError::ConstantBufferSize { bank: 2, size: 8 })
    );
}

/// A read whose bank is a register is refused by name.
#[test]
fn a_register_bank_is_refused() {
    let function = ir("block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetCbufU32 %0, u32 0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   return\n");
    let at = value_at(&function, 1);
    let error = lower_with(GuestStage::Compute, &function, &ONE_INVOCATION).err();
    assert_eq!(error, Some(LowerError::ConstantBufferRegisterBank { at }));
}

/// Each refusal of one read, at its edge and in its order: the arguments' types (the
/// verifier's), the bank's range (31 accepted, 32 refused) and the bank's size; with two
/// faults, the earlier check reports. A register offset is accepted.
#[test]
fn each_read_check_at_its_edge() {
    let read = |bank: &str, offset: &str| {
        ir(&format!(
            "block0:\n\
             \x20   %0: u32 = GetCbufU32 {bank}, {offset}\n\
             \x20   %1 = SetRegister r0, %0\n\
             \x20   return\n"
        ))
    };
    let all_banks = LowerInputs {
        constant_buffers: [CBUF_ALIGNMENT; CONSTANT_BUFFER_SLOTS],
        ..ONE_INVOCATION
    };
    let refused = |function: &Function, inputs: &LowerInputs| {
        lower_with(GuestStage::Compute, function, inputs).err()
    };
    let last = u32::from(CBUF_BANKS) - 1;
    let accepted = read(&format!("u32 {last}"), "u32 0");
    assert_eq!(refused(&accepted, &all_banks), None);
    let past = read(&format!("u32 {}", last + 1), "u32 0");
    let at = value_at(&past, 0);
    assert_eq!(
        refused(&past, &all_banks),
        Some(LowerError::ConstantBufferBank { at, bank: last + 1 })
    );
    let huge = read(&format!("u32 {:#x}", 0x100 + last), "u32 0");
    assert_eq!(
        refused(&huge, &all_banks),
        Some(LowerError::ConstantBufferBank {
            at,
            bank: 0x100 + last
        }),
        "a bank whose low byte is a bank is still refused"
    );
    // An argument that is not a `u32` is the verifier's (step 1).
    let wrong_bank = read("u1 true", "u32 0");
    assert_eq!(
        refused(&wrong_bank, &all_banks),
        Some(LowerError::Verify(VerifyError::Type {
            at: crate::ir::Site::Inst(at),
            index: 0,
            wanted: Type::U32,
            found: Type::U1
        }))
    );
    let wrong_offset = read("u32 0", "u1 true");
    assert_eq!(
        refused(&wrong_offset, &all_banks),
        Some(LowerError::Verify(VerifyError::Type {
            at: crate::ir::Site::Inst(at),
            index: 1,
            wanted: Type::U32,
            found: Type::U1
        }))
    );
    let no_size = read("u32 6", "u32 0");
    assert_eq!(
        refused(&no_size, &with_bank(ONE_INVOCATION, 5, 16)),
        Some(LowerError::ConstantBufferUnsized { at, bank: 6 })
    );
    assert_eq!(refused(&no_size, &with_bank(ONE_INVOCATION, 6, 16)), None);
    // Two faults: the bank's range is checked before its size (bank 32 has no size).
    let past_and_no_size = ir(&format!(
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetCbufU32 u32 {}, %0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   return\n",
        last + 1
    ));
    let read_at = value_at(&past_and_no_size, 1);
    assert_eq!(
        refused(&past_and_no_size, &ONE_INVOCATION),
        Some(LowerError::ConstantBufferBank {
            at: read_at,
            bank: last + 1
        })
    );
}

/// The new errors read as words.
#[test]
fn resource_errors_read_as_words() {
    let at = value_at(&reading(GuestStage::Compute, &[0, 1, 2, 3]), 3);
    for (error, text) in [
        (
            LowerError::ConstantBufferSize { bank: 4, size: 17 },
            "constant buffer bank 4 is given 17 bytes; a bank's size is a multiple of 16, at \
             most 65536",
        ),
        (
            LowerError::ConstantBufferRegisterBank { at },
            "%3: the constant buffer bank is a register, which is not lowered yet",
        ),
        (
            LowerError::ConstantBufferBank { at, bank: 32 },
            "%3: constant buffer bank 32 does not exist; banks stop at 31",
        ),
        (
            LowerError::ConstantBufferUnsized { at, bank: 0 },
            "%3: constant buffer bank 0 is read but given no size",
        ),
    ] {
        assert_eq!(error.to_string(), text);
    }
}
