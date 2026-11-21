//! Local and shared memory: the arrays' sizes, every access at and past the end on the host's
//! GPU, the size caps, and the stages that may use them.

use naga::{AddressSpace, ArraySize};

use super::run::{gpu_registers, opaque};
use super::*;
use crate::lower::memory::words;

/// The length in words and address space of each global array of `shader`'s module.
fn arrays(shader: &HostShader) -> Vec<(String, AddressSpace, u32)> {
    let module = naga_module(shader);
    module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let length = match module.types[global.ty].inner {
                TypeInner::Array {
                    size: ArraySize::Constant(length),
                    ..
                } => Some(length.get()),
                _ => None,
            }
            .expect("a sized array");
            (global.name.clone().expect("named"), global.space, length)
        })
        .collect()
}

/// A compute function that runs `body` (instructions numbered from %0) and then returns.
fn compute(body: &str) -> Function {
    ir(&format!("block0:\n{body}    return\n"))
}

/// The bytes of `words` words.
fn bytes(words: u32) -> u32 {
    words * MEMORY_WORD_BYTES
}

/// The arrays are `ceil(bytes / 4)` words: each size at a word boundary and one byte past it,
/// and none for zero bytes or for a program that does not use the space.
#[test]
fn arrays_are_sized_in_whole_words() {
    assert_eq!(
        [0, 1, 4, 5, 8].map(words),
        [0, 1, 1, 2, 2],
        "Eden's DivCeil(size, 4U)"
    );
    let uses_both = compute("    %0: u32 = LoadLocal u32 0\n    %1: u32 = LoadSharedU32 u32 0\n");
    for (bytes, length) in [(4, 1), (5, 2), (8, 2), (9, 3)] {
        let shader = lowered(
            GuestStage::Compute,
            &uses_both,
            &LowerInputs {
                local_memory_bytes: u64::from(bytes),
                shared_memory_bytes: bytes,
                ..LowerInputs::DEFAULT
            },
        );
        assert_eq!(
            arrays(&shader),
            [
                ("local_memory".to_owned(), AddressSpace::Private, length),
                ("shared_memory".to_owned(), AddressSpace::WorkGroup, length),
            ],
            "{bytes} bytes"
        );
    }
    let sized = LowerInputs {
        local_memory_bytes: 16,
        shared_memory_bytes: 16,
        ..LowerInputs::DEFAULT
    };
    // No array for zero bytes, or when the program does not use the space.
    let none = LowerInputs {
        local_memory_bytes: 0,
        shared_memory_bytes: 0,
        ..LowerInputs::DEFAULT
    };
    assert_eq!(arrays(&lowered(GuestStage::Compute, &uses_both, &none)), []);
    let neither = compute("    %0: u32 = IAdd32 u32 1, u32 2\n");
    assert_eq!(arrays(&lowered(GuestStage::Compute, &neither, &sized)), []);
}

/// What one register of a memory test should hold, and the instructions that compute it.
struct Probe {
    name: &'static str,
    expected: u32,
}

/// Runs `body` (which writes registers r0 up, one per probe) as a compute shader with
/// `inputs`, and compares the registers the GPU leaves with each probe's expected value. The
/// operands are made opaque to the host compiler first. The interpreter refuses an access past
/// the end, so the expected values are written out (see `memory`'s module docs).
fn check_on_gpu(test: &str, body: &str, inputs: &LowerInputs, probes: &[Probe]) {
    let function = opaque(&compute(body));
    let shader = lowered(GuestStage::Compute, &function, inputs);
    let regs: Vec<u8> = (0..u8::try_from(probes.len()).expect("few")).collect();
    let Some(found) = gpu_registers(test, &shader, &regs) else {
        return;
    };
    let wrong: Vec<String> = probes
        .iter()
        .zip(&found)
        .filter(|(probe, found)| probe.expected != **found)
        .map(|(probe, found)| {
            format!(
                "{}: expected {:#x}, GPU {found:#x}",
                probe.name, probe.expected
            )
        })
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Local memory of three words: a write and a read at the last word (index 2) land, a write at
/// index 3 (the first past the end) is dropped and a read there gives 0, as does a read at
/// `u32::MAX`; the neighbour at index 1 keeps its own value.
#[test]
fn local_accesses_at_and_past_the_end() {
    let length: u32 = 3;
    let last = length - 1;
    let body = format!(
        "    %0 = WriteLocal u32 {before}, u32 0x11
    %1 = WriteLocal u32 {last}, u32 0x22
    %2 = WriteLocal u32 {length}, u32 0x33
    %3: u32 = LoadLocal u32 {before}
    %4: u32 = LoadLocal u32 {last}
    %5: u32 = LoadLocal u32 {length}
    %6: u32 = LoadLocal u32 {max}
    %7 = SetRegister r0, %3
    %8 = SetRegister r1, %4
    %9 = SetRegister r2, %5
    %10 = SetRegister r3, %6
",
        before = last - 1,
        max = u32::MAX
    );
    // One byte short of three words still makes three.
    let inputs = LowerInputs {
        local_memory_bytes: u64::from(bytes(length) - 1),
        ..LowerInputs::DEFAULT
    };
    check_on_gpu(
        "local_accesses_at_and_past_the_end",
        &body,
        &inputs,
        &[
            Probe {
                name: "the neighbour",
                expected: 0x11,
            },
            Probe {
                name: "the last word",
                expected: 0x22,
            },
            Probe {
                name: "the first word past the end",
                expected: 0,
            },
            Probe {
                name: "u32::MAX",
                expected: 0,
            },
        ],
    );
}

/// With no local memory, every read is 0 and every write is dropped, and no array is made.
#[test]
fn empty_local_memory_reads_zero() {
    let body = "    %0 = WriteLocal u32 0, u32 0x11
    %1: u32 = LoadLocal u32 0
    %2 = SetRegister r0, %1
";
    let shader = lowered(GuestStage::Pixel, &compute(body), &LowerInputs::DEFAULT);
    assert_eq!(arrays(&shader), []);
    check_on_gpu(
        "empty_local_memory_reads_zero",
        body,
        &LowerInputs::DEFAULT,
        &[Probe {
            name: "the only word",
            expected: 0,
        }],
    );
}

/// Shared memory of four words (16 bytes), by byte offset: 32-, 64- and 128-bit stores and
/// loads at and past the end (a 64-bit access straddling the end keeps its word inside), the
/// 128-bit store's fourth word read back before the stores at the last word overwrite it, the
/// 8- and 16-bit loads of each field, and the offset whose word index Eden's arithmetic shift
/// would wrap to word 0.
#[test]
fn shared_offsets_at_and_past_the_end() {
    let length: u32 = 4;
    let end = bytes(length);
    let last = end - MEMORY_WORD_BYTES;
    let body = format!(
        "    %0: u32x4 = CompositeConstructU32x4 u32 0x84838281, u32 0x2, u32 0x3, u32 0x4
    %1 = WriteSharedU128 u32 0, %0
    %2: u32 = LoadSharedU32 u32 {last}
    %3 = WriteSharedU32 u32 {last}, u32 0xf00d
    %4 = WriteSharedU32 u32 {end}, u32 0xdead
    %5: u32x2 = CompositeConstructU32x2 u32 0x55, u32 0x66
    %6 = WriteSharedU64 u32 {last}, %5
    %7: u32 = LoadSharedU32 u32 {last}
    %8: u32 = LoadSharedU32 u32 {end}
    %9: u32x2 = LoadSharedU64 u32 {last}
    %10: u32 = CompositeExtractU32x2 %9, u32 0
    %11: u32 = CompositeExtractU32x2 %9, u32 1
    %12: u32x4 = LoadSharedU128 u32 0
    %13: u32 = CompositeExtractU32x4 %12, u32 1
    %14: u32 = CompositeExtractU32x4 %12, u32 2
    %15: u32 = LoadSharedU8 u32 3
    %16: u32 = LoadSharedS8 u32 3
    %17: u32 = LoadSharedU16 u32 2
    %18: u32 = LoadSharedS16 u32 2
    %19: u32 = LoadSharedS8 u32 0
    %20: u32x2 = LoadSharedU64 u32 0xfffffffc
    %21: u32 = CompositeExtractU32x2 %20, u32 1
    %22 = SetRegister r0, %7
    %23 = SetRegister r1, %8
    %24 = SetRegister r2, %10
    %25 = SetRegister r3, %11
    %26 = SetRegister r4, %13
    %27 = SetRegister r5, %14
    %28 = SetRegister r6, %15
    %29 = SetRegister r7, %16
    %30 = SetRegister r8, %17
    %31 = SetRegister r9, %18
    %32 = SetRegister r10, %19
    %33 = SetRegister r11, %21
    %34 = SetRegister r12, %2
"
    );
    let inputs = LowerInputs {
        shared_memory_bytes: end,
        ..LowerInputs::DEFAULT
    };
    let probe = |name, expected| Probe { name, expected };
    check_on_gpu(
        "shared_offsets_at_and_past_the_end",
        &body,
        &inputs,
        &[
            // The 64-bit store at the last word wrote its first half there.
            probe("the last word", 0x55),
            probe("the first word past the end", 0),
            probe("a 64-bit load at the last word, first half", 0x55),
            probe("a 64-bit load at the last word, second half", 0),
            probe("a 128-bit load, word 1", 0x2),
            probe("a 128-bit load, word 2", 0x3),
            probe("byte 3, unsigned", 0x84),
            probe("byte 3, signed", 0xffff_ff84),
            probe("half 1, unsigned", 0x8483),
            probe("half 1, signed", 0xffff_8483),
            probe("byte 0, signed", 0xffff_ff81),
            probe("the word after index 0x3fff_ffff, not word 0", 0),
            // Read before the 32- and 64-bit stores at the last word overwrite it.
            probe("a 128-bit store at 0, word 3", 0x4),
        ],
    );
}

/// Shared memory is a compute program's: refused in the other stages (the atomics and narrow
/// stores too), and the narrow stores and the atomics are not lowered in compute programs.
#[test]
fn shared_memory_only_in_compute() {
    for text in [
        "    %0: u32 = LoadSharedU32 u32 0\n",
        "    %0 = WriteSharedU8 u32 0, u32 1\n",
        "    %0: u32 = SharedAtomicIAdd32 u32 0, u32 1\n",
    ] {
        let function = compute(text);
        let opcode = match function.inst(value_at(&function, 0)) {
            Some(Inst::Op(op)) => Some(op.opcode),
            _ => None,
        }
        .expect("an operation");
        for stage in [GuestStage::Vertex, GuestStage::Pixel] {
            assert_eq!(
                lower(stage, &function).err(),
                Some(LowerError::NotInStage {
                    at: value_at(&function, 0),
                    opcode,
                    stage,
                }),
                "{text}"
            );
        }
        let in_compute = lower(GuestStage::Compute, &function);
        if opcode == Opcode::LoadSharedU32 {
            assert!(in_compute.is_ok());
        } else {
            assert_eq!(
                in_compute.err(),
                Some(LowerError::Operation {
                    at: value_at(&function, 0),
                    opcode
                })
            );
        }
    }
    // `WriteSharedU16` too.
    let function = compute("    %0 = WriteSharedU16 u32 0, u32 1\n");
    assert_eq!(
        lower(GuestStage::Compute, &function).err(),
        Some(LowerError::Operation {
            at: value_at(&function, 0),
            opcode: Opcode::WriteSharedU16
        })
    );
}

/// The size caps at their edges: [`MAX_LOCAL_MEMORY_BYTES`] in every stage and
/// [`MAX_SHARED_MEMORY_BYTES`] in compute programs, accepted at the cap and refused one byte
/// past it; the shared size is ignored outside compute.
#[test]
fn memory_caps_at_their_edges() {
    let function = compute("    %0: u32 = LoadLocal u32 0\n");
    for stage in [GuestStage::Vertex, GuestStage::Pixel, GuestStage::Compute] {
        let at = |bytes| {
            lower_with(
                stage,
                &function,
                &LowerInputs {
                    local_memory_bytes: bytes,
                    ..LowerInputs::DEFAULT
                },
            )
        };
        let shader = at(MAX_LOCAL_MEMORY_BYTES).expect("at the cap");
        let length =
            u32::try_from(MAX_LOCAL_MEMORY_BYTES / u64::from(MEMORY_WORD_BYTES)).expect("small");
        assert_eq!(
            arrays(&shader),
            [("local_memory".to_owned(), AddressSpace::Private, length)]
        );
        assert_eq!(
            at(MAX_LOCAL_MEMORY_BYTES + 1).err(),
            Some(LowerError::LocalMemory {
                bytes: MAX_LOCAL_MEMORY_BYTES + 1
            })
        );
        assert_eq!(
            at(u64::MAX).err(),
            Some(LowerError::LocalMemory { bytes: u64::MAX })
        );
    }
    let shared = compute("    %0: u32 = LoadSharedU32 u32 0\n");
    let at = |bytes| {
        lower_with(
            GuestStage::Compute,
            &shared,
            &LowerInputs {
                shared_memory_bytes: bytes,
                ..LowerInputs::DEFAULT
            },
        )
    };
    assert!(at(MAX_SHARED_MEMORY_BYTES).is_ok());
    assert_eq!(
        at(MAX_SHARED_MEMORY_BYTES + 1).err(),
        Some(LowerError::SharedMemory {
            bytes: MAX_SHARED_MEMORY_BYTES + 1
        })
    );
    let pixel = lower_with(
        GuestStage::Pixel,
        &ir("block0:\n    return\n"),
        &LowerInputs {
            shared_memory_bytes: u32::MAX,
            ..LowerInputs::DEFAULT
        },
    );
    assert!(pixel.is_ok());
}

/// The MSL naga writes for a compute program's local and shared accesses (snapshot; a change
/// is explained in its PR).
#[test]
fn memory_snapshot() {
    let function = compute(
        "    %0 = WriteLocal u32 1, u32 7
    %1: u32 = LoadLocal u32 1
    %2: u32x2 = LoadSharedU64 u32 4
    %3: u32 = LoadSharedS16 u32 2
    %4 = WriteSharedU32 u32 8, %1
",
    );
    let shader = lowered(
        GuestStage::Compute,
        &function,
        &LowerInputs {
            local_memory_bytes: 8,
            shared_memory_bytes: 12,
            ..LowerInputs::DEFAULT
        },
    );
    insta::assert_snapshot!("memory", msl(&shader));
}

/// `LDL` or `STL` of one word at byte `offset` (`[RZ + offset]`).
fn local_access(load: bool, reg: u8, offset: u32) -> crate::decode::Instr {
    local_access_with(
        load,
        reg,
        crate::decode::Gpr::RZ,
        offset,
        crate::decode::MemType::B32,
    )
}

/// `LDL` or `STL` of `mem_type` at `[base + offset]`, as the translator takes them; `LDL`
/// encodes the uniform register field (register 0, off), as PR #4588 takes it.
fn local_access_with(
    load: bool,
    reg: u8,
    base: crate::decode::Gpr,
    offset: u32,
    mem_type: crate::decode::MemType,
) -> crate::decode::Instr {
    use crate::decode::{
        Gpr, Instr, MemAddr, MemEvictionPriority, MemSpace, Op, OpLd, OpSt, Pred, PredSrc,
        UniformAddr, UniformReg,
    };
    let addr = MemAddr {
        reg: base,
        reg_64: false,
        uniform: None,
        offset: i32::try_from(offset).expect("small"),
    };
    let space = MemSpace::Local(MemEvictionPriority::Normal);
    if load {
        Instr::new(Op::Ld(OpLd {
            dst: Gpr::new(reg),
            addr: MemAddr {
                uniform: Some(UniformAddr {
                    reg: UniformReg::new(0).expect("a uniform register"),
                    wide: false,
                    enabled: false,
                }),
                ..addr
            },
            mem_type,
            space,
            pred: PredSrc::TRUE,
            pred_dst: Pred::PT,
            zd: false,
            bit76: false,
        }))
    } else {
        Instr::new(Op::St(OpSt {
            data: Gpr::new(reg),
            addr,
            mem_type,
            space,
            bit76: false,
        }))
    }
}

/// `program` (with `MOV`s of `values` into their registers first and `EXIT` last), translated
/// by the translation driver.
fn translated_local(values: &[(u8, u32)], program: &[crate::decode::Instr]) -> Function {
    use crate::decode::{Gpr, Instr, MovSrc, Op, OpExit, OpMov, PredSrc};
    use crate::translate::driver::translate_instrs;
    use crate::translate::{COLOR_TARGETS, ColorOutputs, Stage};
    let mut code: Vec<Instr> = values
        .iter()
        .map(|&(reg, bits)| {
            Instr::new(Op::Mov(OpMov {
                dst: Gpr::new(reg),
                src: MovSrc::Imm(bits),
                quad_lanes: OpMov::ALL_LANES,
            }))
        })
        .collect();
    code.extend_from_slice(program);
    code.push(Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    })));
    // The translator takes vertex and pixel programs; a vertex program's exit adds nothing
    // the compute lowering refuses.
    let stage = Stage {
        stage: GuestStage::Vertex,
        color_outputs: ColorOutputs {
            targets: [0; COLOR_TARGETS],
        },
    };
    translate_instrs(stage, None, &code).expect("translates")
}

/// The `n`th `opcode` instruction `function` lists.
fn nth_listed(function: &Function, opcode: Opcode, n: usize) -> crate::ir::Value {
    function
        .listed()
        .filter(|(_, inst)| matches!(inst, crate::ir::Inst::Op(op) if op.opcode == opcode))
        .nth(n)
        .map(|(value, _)| value)
        .expect("listed")
}

/// Translated `STL` and `LDL` against the interpreter, with local memory of three words on
/// both: at the last word (byte 8) the GPU gives what the interpreter gives; one word past it
/// (byte 12, the size the lowering was given) the interpreter refuses the access with its exact
/// error, at the access itself, and the GPU reads 0 for the load and drops the store, leaving
/// the last word's value (the divergence `lower::memory` lists), for a store at an immediate
/// offset and at a register offset. Eden's translator selects 0 for an `LDL` whose byte offset
/// is not below `LocalMemorySize()`, and drops an `STL` at an immediate offset not below it
/// (`load_store_local_shared.cpp`), so for those accesses the GPU's results are also Eden's;
/// Eden does not bound the register-offset store (`lower::memory`'s divergences).
#[test]
fn translated_local_accesses_match_the_interpreter_at_the_edge() {
    let length: u32 = 3;
    let last = bytes(length - 1);
    let end = bytes(length);
    let inputs = LowerInputs {
        local_memory_bytes: u64::from(end),
        ..LowerInputs::DEFAULT
    };
    let interp = |function: &Function| {
        let mut machine = crate::interp::Machine::new();
        machine.local = vec![0; usize::try_from(length).expect("small")];
        machine.run(function).map(|_| machine.regs)
    };
    let gpu = |test: &str, function: &Function, regs: &[u8]| {
        let shader = lowered(GuestStage::Compute, &opaque(function), &inputs);
        gpu_registers(test, &shader, regs)
    };
    let out_of_bounds = |at| crate::interp::InterpError::OutOfBounds {
        at,
        space: crate::interp::Space::Local,
        offset: u64::from(end),
        size: u64::from(MEMORY_WORD_BYTES),
        len: u64::from(end),
    };

    // At the last word: both sides agree.
    let inside = translated_local(
        &[(0, 0x22)],
        &[local_access(false, 0, last), local_access(true, 1, last)],
    );
    let regs = interp(&inside).expect("in bounds");
    assert_eq!(regs[1], 0x22, "the interpreter");
    if let Some(found) = gpu("translated_local_inside", &inside, &[1]) {
        assert_eq!(found, [regs[1]], "the GPU");
    }

    // A load one word past the end.
    let load_past = translated_local(
        &[(0, 0x11)],
        &[
            local_access(false, 0, last),
            local_access(true, 1, end),
            local_access(true, 3, last),
        ],
    );
    assert_eq!(
        interp(&load_past).map(|_| ()),
        Err(out_of_bounds(nth_listed(&load_past, Opcode::LoadLocal, 0)))
    );
    if let Some(found) = gpu("translated_local_load_past", &load_past, &[1, 3]) {
        assert_eq!(found, [0, 0x11], "the GPU");
    }

    // A store one word past the end.
    let store_past = translated_local(
        &[(0, 0x11), (2, 0x33)],
        &[
            local_access(false, 0, last),
            local_access(false, 2, end),
            local_access(true, 3, last),
        ],
    );
    assert_eq!(
        interp(&store_past).map(|_| ()),
        Err(out_of_bounds(nth_listed(
            &store_past,
            Opcode::WriteLocal,
            1
        )))
    );
    if let Some(found) = gpu("translated_local_store_past", &store_past, &[3]) {
        assert_eq!(found, [0x11], "the GPU");
    }

    // A store one word past the end at a register offset (`STL [R4]`, R4 = 12).
    let register_past = translated_local(
        &[(0, 0x11), (2, 0x33), (4, end)],
        &[
            local_access(false, 0, last),
            local_access_with(
                false,
                2,
                crate::decode::Gpr::new(4),
                0,
                crate::decode::MemType::B32,
            ),
            local_access(true, 3, last),
        ],
    );
    assert_eq!(
        interp(&register_past).map(|_| ()),
        Err(out_of_bounds(nth_listed(
            &register_past,
            Opcode::WriteLocal,
            1
        )))
    );
    if let Some(found) = gpu("translated_local_register_past", &register_past, &[3]) {
        assert_eq!(found, [0x11], "the GPU");
    }
}

/// Divergence (`lower::memory`): with a local-memory size that is not a multiple of 4 (13
/// bytes, four words), a translated `LDL.U8` at byte 13, in the last word but at the byte size,
/// reads the stored byte here, on the interpreter and on the GPU alike (0xcc of 0xaabb_ccdd),
/// where Eden's translator selects 0 (`ILessThan(13, 13)` is false). The store at byte 12 is
/// below the size, so Eden keeps it too.
#[test]
fn divergence_a_narrow_load_past_the_byte_size_in_the_last_word_reads_it() {
    let size: u32 = 13;
    let words = u32::try_from(words(u64::from(size))).expect("small");
    assert_eq!(words, 4);
    let last = bytes(words - 1);
    let function = translated_local(
        &[(0, 0xaabb_ccdd)],
        &[
            local_access(false, 0, last),
            local_access_with(
                true,
                1,
                crate::decode::Gpr::RZ,
                size,
                crate::decode::MemType::U8,
            ),
        ],
    );
    let mut machine = crate::interp::Machine::new();
    machine.local = vec![0; usize::try_from(words).expect("small")];
    machine.run(&function).expect("in bounds by word");
    assert_eq!(machine.regs[1], 0xcc, "the interpreter");
    let inputs = LowerInputs {
        local_memory_bytes: u64::from(size),
        ..LowerInputs::DEFAULT
    };
    let shader = lowered(GuestStage::Compute, &opaque(&function), &inputs);
    if let Some(found) = gpu_registers(
        "divergence_a_narrow_load_past_the_byte_size_in_the_last_word_reads_it",
        &shader,
        &[1],
    ) {
        assert_eq!(found, [0xcc], "the GPU");
    }
}
