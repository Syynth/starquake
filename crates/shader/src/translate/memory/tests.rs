//! Tests of the memory translator: each instruction, space, type, atomic operation and
//! modifier run through the IR interpreter (slice 7) with expected values from the
//! instruction's documented behaviour, golden IR, the guards, every refusal with its exact
//! error, PR #4588's cases for this group and a no-panic property test.
//!
//! Every case is built as an [`Instr`], assembled by the test assembler and decoded again
//! before it is translated, so the words are synthetic (never copied from a game).

mod atomics;
mod barriers;
mod constant;
mod global;
mod golden;
mod guards;
mod local;
mod pr4588;
mod rejections;
mod shared;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{
    CBufRef, GlobalAccess, MemEvictionPriority, OpCCtl, OpErrBar, OpLdgDepBar, PredSrc, UniformAddr,
};
use crate::interp::{InterpError, Machine, Region};
use crate::ir::{Function, Terminator};
use crate::testing::{any_word_of, config};

pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

pub(super) fn p(index: u8) -> Pred {
    Pred::new(index).unwrap_or(Pred::PT)
}

pub(super) fn ps(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated,
    }
}

pub(super) fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).unwrap_or(UniformReg::URZ)
}

/// A weak global access with the normal eviction priority (NAK's defaults).
pub(super) const GLOBAL: MemSpace = MemSpace::Global(GlobalAccess {
    order: MemOrder::Weak,
    eviction: MemEvictionPriority::Normal,
});

/// A global access with `order`.
pub(super) const fn global_with(order: MemOrder) -> MemSpace {
    MemSpace::Global(GlobalAccess {
        order,
        eviction: MemEvictionPriority::Normal,
    })
}

/// The local space with the normal eviction priority.
pub(super) const LOCAL: MemSpace = MemSpace::Local(MemEvictionPriority::Normal);

/// An address of the forms without a uniform register: `reg` (a pair when `pair`) plus
/// `offset`.
pub(super) fn plain(reg: u8, pair: bool, offset: i32) -> MemAddr {
    MemAddr {
        reg: r(reg),
        reg_64: pair,
        uniform: None,
        offset,
    }
}

/// An address of the uniform forms: `reg` (a pair when `reg_64`) plus `offset`, with the
/// uniform register `uniform` (a pair when `wide`), on when `enabled`.
pub(super) fn with_uniform(
    reg: u8,
    reg_64: bool,
    uniform: u8,
    wide: bool,
    enabled: bool,
    offset: i32,
) -> MemAddr {
    MemAddr {
        reg: r(reg),
        reg_64,
        uniform: Some(UniformAddr {
            reg: ur(uniform),
            wide,
            enabled,
        }),
        offset,
    }
}

/// The address of `LDS` and `LDL` as PR #4588 takes it: `reg` plus `offset`, the uniform
/// register field 0 and off.
pub(super) fn ld_addr(reg: u8, offset: i32) -> MemAddr {
    with_uniform(reg, false, 0, false, false, offset)
}

/// A global load of `mem_type` into `dst` from `addr`.
pub(super) fn ldg(dst: u8, mem_type: MemType, addr: MemAddr) -> OpLd {
    OpLd {
        dst: r(dst),
        addr,
        mem_type,
        space: GLOBAL,
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }
}

/// A shared load of `mem_type` into `dst` from `reg` scaled by `stride`, plus `offset`.
pub(super) fn lds(dst: u8, mem_type: MemType, reg: u8, stride: OffsetStride, offset: i32) -> OpLd {
    OpLd {
        space: MemSpace::Shared(stride),
        ..ldg(dst, mem_type, ld_addr(reg, offset))
    }
}

/// A local load of `mem_type` into `dst` from `reg` plus `offset`.
pub(super) fn ldl(dst: u8, mem_type: MemType, reg: u8, offset: i32) -> OpLd {
    OpLd {
        space: LOCAL,
        ..ldg(dst, mem_type, ld_addr(reg, offset))
    }
}

/// A store of `mem_type` from `data` to `addr` in `space`.
pub(super) fn st(data: u8, mem_type: MemType, space: MemSpace, addr: MemAddr) -> OpSt {
    OpSt {
        data: r(data),
        addr,
        mem_type,
        space,
        bit76: false,
    }
}

/// An atomic `op` of `atom_type` in `space` at `addr`, data `data`, old value to `dst`.
pub(super) fn atom(
    dst: u8,
    op: AtomOp,
    atom_type: AtomType,
    space: MemSpace,
    addr: MemAddr,
    data: u8,
) -> OpAtom {
    OpAtom {
        dst: AtomDst::Reg(r(dst)),
        addr,
        data: r(data),
        op,
        atom_type,
        space,
        pred_dst: Pred::PT,
    }
}

/// A `RED` (`op` of `atom_type`) at `reg` (a pair when `pair`, bit 72) plus `offset`, as PR
/// #4588 takes it: the destination and uniform fields 0, bits 90 and 91 clear.
pub(super) fn red(
    op: AtomOp,
    atom_type: AtomType,
    reg: u8,
    pair: bool,
    offset: i32,
    data: u8,
) -> OpAtom {
    OpAtom {
        dst: AtomDst::Reduction { ignored: r(0) },
        ..atom(
            0,
            op,
            atom_type,
            GLOBAL,
            with_uniform(reg, false, 0, pair, false, offset),
            data,
        )
    }
}

/// `instr` round-tripped through its word: assembled, then decoded.
pub(super) fn word(instr: &Instr) -> Instruction {
    let word = assemble(instr).expect("the test assembler encodes the case");
    assert_eq!(decode(word), Ok(*instr), "the case decodes back to itself");
    word
}

/// The function `instrs` translate to, through their words, in one entry block that returns.
pub(super) fn build(instrs: &[Instr]) -> Result<Function, MemoryError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for instr in instrs {
        translate_memory_word(&mut builder, word(instr), GuardMode::Honour)?;
    }
    builder.terminate(Terminator::Return)?;
    Ok(function)
}

/// The function one unguarded operation translates to.
pub(super) fn build_op(op: Op) -> Result<Function, MemoryError> {
    build(&[Instr::new(op)])
}

/// Translates `instrs` and runs them on `machine`.
pub(super) fn run(instrs: &[Instr], machine: &mut Machine) {
    let function = build(instrs).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    machine.run(&function).expect("runs");
}

/// Runs one unguarded operation on `machine`.
pub(super) fn run_op(op: Op, machine: &mut Machine) {
    run(&[Instr::new(op)], machine);
}

/// A machine with `regs` set from register 0 up.
pub(super) fn machine_with(regs: &[u32]) -> Machine {
    let mut machine = Machine::new();
    for (slot, value) in machine.regs.iter_mut().zip(regs) {
        *slot = *value;
    }
    machine
}

/// Sets register `reg` of `machine`.
pub(super) fn set_reg(machine: &mut Machine, reg: u8, value: u32) {
    if let Some(slot) = machine.regs.get_mut(usize::from(reg)) {
        *slot = value;
    }
}

/// Sets uniform register `reg` of `machine`.
pub(super) fn set_uniform(machine: &mut Machine, reg: u8, value: u32) {
    if let Some(slot) = machine.uniform_regs.get_mut(usize::from(reg)) {
        *slot = value;
    }
}

/// Global memory of `len` bytes at `base`, every byte `fill`.
pub(super) fn global_region(machine: &mut Machine, base: u64, len: usize, fill: u8) {
    machine.global = Region {
        base,
        bytes: vec![fill; len],
    };
}

/// The 32-bit word at global address `address`.
pub(super) fn global_word(machine: &Machine, address: u64) -> u32 {
    let start = address
        .checked_sub(machine.global.base)
        .and_then(|start| usize::try_from(start).ok())
        .expect("inside the region");
    let bytes = machine.global.bytes[start..start + 4]
        .try_into()
        .expect("four bytes");
    u32::from_le_bytes(bytes)
}

/// Writes the 32-bit word `value` at global address `address`.
pub(super) fn put_global_word(machine: &mut Machine, address: u64, value: u32) {
    let start = address
        .checked_sub(machine.global.base)
        .and_then(|start| usize::try_from(start).ok())
        .expect("inside the region");
    machine.global.bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
}

/// The 32-bit word at shared byte address `address`.
pub(super) fn shared_word(machine: &Machine, address: usize) -> u32 {
    let bytes = machine.shared[address..address + 4]
        .try_into()
        .expect("four bytes");
    u32::from_le_bytes(bytes)
}

/// Writes the 32-bit word `value` at shared byte address `address`.
pub(super) fn put_shared_word(machine: &mut Machine, address: usize, value: u32) {
    machine.shared[address..address + 4].copy_from_slice(&value.to_le_bytes());
}

/// Constant buffer `bank` holding `words` from offset 0, little-endian.
pub(super) fn with_cbuf(machine: &mut Machine, bank: u32, words: &[u32]) {
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine.cbufs.insert(bank, bytes);
}

/// `value`'s bits as an unsigned integer.
pub(super) fn to_u32(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

/// `value` sign-extended to 64 bits, as an unsigned integer.
pub(super) fn sext64(value: i32) -> u64 {
    u64::from_ne_bytes(i64::from(value).to_ne_bytes())
}

/// The values the generated cases draw from: the edges of the 32-bit range, and two patterns
/// with every byte distinct.
pub(super) const VALUES: [u32; 7] = [
    0,
    1,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x1234_5678,
    0x8765_4321,
];

/// Whether `error` is one an interpreter run of a translated memory instruction may end with
/// on arbitrary registers: an access outside or misaligned in the memory the machine holds.
fn memory_fault(error: &InterpError) -> bool {
    matches!(
        error,
        InterpError::OutOfBounds { .. } | InterpError::Misaligned { .. }
    )
}

proptest! {
    #![proptest_config(config())]

    /// Generated words of every memory form decode or not; what decodes translates or is
    /// refused, never panics; a refusal leaves the function as it was; what translates
    /// verifies and runs on the interpreter, stopping at most at a memory fault (the
    /// registers address arbitrary memory).
    #[test]
    fn generated_words_never_panic(
        word in any_word_of(crate::decode::memory::FORMS),
        regs in proptest::collection::vec(any::<u32>(), 256),
        uniform_regs in proptest::collection::vec(any::<u32>(), 64),
        preds in any::<[bool; 8]>(),
        guard in any::<bool>(),
    ) {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let mode = if guard { GuardMode::Honour } else { GuardMode::Tested };
        let before = builder.function().clone();
        match translate_memory_word(&mut builder, word, mode) {
            Ok(()) => {
                prop_assert!(builder.terminate(Terminator::Return).is_ok());
                prop_assert_eq!(function.verify(), Ok(()));
                let mut machine = Machine::new();
                for (slot, value) in machine.regs.iter_mut().zip(&regs) {
                    *slot = *value;
                }
                for (slot, value) in machine.uniform_regs.iter_mut().zip(&uniform_regs) {
                    *slot = *value;
                }
                machine.preds = preds;
                machine.shared = vec![0x5a; 0x100];
                machine.local = vec![0x5a5a_5a5a; 0x40];
                for bank in 0..32 {
                    machine.cbufs.insert(bank, vec![0xa5; 0x100]);
                }
                if let Err(error) = machine.run(&function) {
                    prop_assert!(memory_fault(&error), "{error}");
                }
            }
            Err(MemoryError::Build(error)) => prop_assert!(false, "build error {error}"),
            Err(_) => prop_assert_eq!(builder.function(), &before),
        }
    }
}

/// An instruction of another group is refused by its mnemonic, and so is a word that does not
/// decode, with the decoder's exact error.
#[test]
fn other_groups_and_unknown_words_are_refused() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let nop = Instr::new(Op::Nop(crate::decode::OpNop {}));
    let error = translate_memory(&mut builder, &nop, GuardMode::Honour);
    assert_eq!(error, Err(MemoryError::OtherGroup("nop")));
    assert_eq!(
        error.map_err(|error| error.to_string()),
        Err("nop is not a memory instruction".to_owned())
    );
    let unknown = Instruction::from_bits(0);
    assert_eq!(
        translate_memory_word(&mut builder, unknown, GuardMode::Honour),
        Err(MemoryError::Decode(DecodeError::UnknownOpcode {
            opcode: 0
        }))
    );
    assert_eq!(builder.function(), &Function::new());
}

/// [`is_memory`] holds for exactly the group's operations: every decoded form of the memory
/// table, and none of another group's forms.
#[test]
fn the_group_is_the_memory_table() {
    let memory = crate::decode::memory::FORMS;
    let mut seen = 0;
    for group in crate::decode::GROUPS {
        for form in group {
            let ours = memory.iter().any(|row| row.opcode == form.opcode);
            let word = Instruction::from_bits(form.fixed)
                .with_field(crate::decode::OPCODE, u64::from(form.opcode));
            if let Ok(instr) = decode(word) {
                assert_eq!(is_memory(&instr.op), ours, "{}", instr.op.mnemonic());
                seen += usize::from(ours);
            }
        }
    }
    // Every memory form decodes with its fields zero.
    assert_eq!(seen, memory.len());
}

/// `CCTL`, which neither reference translates, is refused by name
/// ([`MemoryUnsupported::CacheControl`]); `ERRBAR` and `LDGDEPBAR` are this group's and
/// translate.
#[test]
fn cctl_is_refused_by_name() {
    let cctl = Op::CCtl(OpCCtl {
        op: crate::decode::CCtlOp::Iv,
        addr: r(2),
        offset: 0x10,
    });
    let refused = build_op(cctl);
    assert_eq!(
        refused,
        Err(MemoryError::Unsupported(MemoryUnsupported::CacheControl))
    );
    assert_eq!(
        refused.map_err(|error| error.to_string()),
        Err("a cctl, which neither reference translates, is not supported".to_owned())
    );
    assert!(build_op(Op::ErrBar(OpErrBar {})).is_ok());
    assert!(build_op(Op::LdgDepBar(OpLdgDepBar {})).is_ok());
}

/// The constants this module mirrors, written out as PR #4588's literals (`ops_memory.cpp`,
/// `ops_misc.cpp` @ e5fb3b97), since a test that imports a constant cannot catch a wrong value.
#[test]
fn constants_match_the_reference_literals() {
    // `dest + 1 < 255`, `uniform + 1 < 63`.
    assert_eq!(REGISTER_END, 255);
    assert_eq!(UNIFORM_REGISTER_END, 63);
    // `Visitor::LDC`: `dest < 254`, `dest < 252`.
    assert_eq!(LDC_PAIR_LIMIT, 254);
    assert_eq!(LDC_QUAD_LIMIT, 252);
    // `Visitor::LDL`: `Imm32(k * 4)`, `ShiftRightArithmetic(.., Imm32(2))`,
    // `ShiftLeftLogical(offset, Imm32(3))`, `Imm32(24)`, `Imm32(16)`.
    assert_eq!(WORD_BYTES, 4);
    assert_eq!(LOCAL_WORD_SHIFT, 2);
    assert_eq!(BYTE_BIT_SHIFT, 3);
    assert_eq!(BYTE_POSITION_MASK, 24);
    assert_eq!(HALF_POSITION_MASK, 16);
    // `Visitor::LDGSTS`: `* 16`, `component < 4`.
    assert_eq!(COPY_SHARED_UNIT, 16);
    assert_eq!(COPY_WORDS, 4);
    // `Visitor::SharedOffset`: the 24-bit field `Take(40, 24)`.
    assert_eq!(OFFSET_FIELD, 0x00ff_ffff);
    // `ShiftLeftLogical(base, Imm32(stride + 1))` for stride codes 1 to 3.
    assert_eq!(stride_shift(OffsetStride::X1), None);
    assert_eq!(stride_shift(OffsetStride::X4), Some(2));
    assert_eq!(stride_shift(OffsetStride::X8), Some(3));
    assert_eq!(stride_shift(OffsetStride::X16), Some(4));
}

/// A constant-buffer reference of `bank` at `offset` (for `LDC`).
pub(super) const fn cbuf(bank: u8, offset: u16) -> CBufRef {
    CBufRef {
        buf: CBuf::Binding(bank),
        offset,
    }
}
