//! Field positions every instruction group shares, written once for the decoder and the test
//! assembler: the opcode, guard, destination and scheduling fields, and the operand slots of
//! ALU instructions. A group's own fields live in its module under `op`.
//!
//! Ported from `mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762 (MIT,
//! Copyright 2022 Collabora, Ltd.): the positions below are those NAK's encoder writes, named
//! after the `SM70Encoder` helper that writes them.

use super::instruction::Field;
use super::operand::{Gpr, HalfSwizzle, Pred, UniformReg};

/// The 12-bit opcode of every instruction (`set_opcode`: bits 0..12). PR #4588 indexes its
/// decode table by these twelve bits (`frontend/ampere/isa.h`, `IsaTable`).
pub(crate) const OPCODE: Field = Field::new(0, 12);
/// The operand form of an ALU instruction (`encode_alu_base`: `set_field(9..12, form)`), the
/// top three bits of the opcode.
pub(crate) const ALU_FORM: Field = Field::new(9, 3);
/// The guard predicate register (`set_pred`: `set_pred_reg(12..15, ..)`).
pub(crate) const GUARD: Field = Field::new(12, Pred::BITS);
/// The guard predicate's negation (`set_pred`: `set_bit(15, pred_inv)`).
pub(crate) const GUARD_NOT: Field = Field::bit(15);
/// The destination register (`set_dst`: bits 16..24).
pub(crate) const DST: Field = Field::new(16, Gpr::BITS);
/// Scheduling information: stall count, yield, barriers, reuse (`set_instr_deps`: bits
/// 105..126 before SM 12.0). Every PR #4588 row marks bits 105..128 known (`isa.h`,
/// `FirstUnknownBit`), so this mask covers through bit 127.
pub(crate) const SCHEDULING: Field = Field::new(105, 23);

/// ALU source 0, always a register (`encode_alu_src0`: `set_alu_reg(24..32, 73, 72, ..)`).
pub(crate) const SRC0: Field = Field::new(24, Gpr::BITS);
/// ALU source 0's absolute-value bit.
pub(crate) const SRC0_ABS: Field = Field::bit(73);
/// ALU source 0's negation bit.
pub(crate) const SRC0_NEG: Field = Field::bit(72);
/// The register in the wide slot, operand form 1 (`encode_alu_reg`: `set_alu_reg(32..40, 62,
/// 63, ..)`).
pub(crate) const SLOT_REG: Field = Field::new(32, Gpr::BITS);
/// The wide slot's absolute-value bit, for a register, uniform register or constant buffer
/// (`encode_alu_reg`, `encode_alu_ureg`, `encode_alu_cb`: `set_bit(62, abs)`).
pub(crate) const SLOT_ABS: Field = Field::bit(62);
/// The wide slot's negation bit (`set_bit(63, neg)` in the same three helpers).
pub(crate) const SLOT_NEG: Field = Field::bit(63);
/// The 32-bit immediate in the wide slot, operand forms 2 and 4 (`encode_alu_imm`:
/// `set_field(32..64, imm)`).
pub(crate) const SLOT_IMM: Field = Field::new(32, 32);
/// The uniform register in the wide slot: the source of operand forms 6 and 7
/// (`encode_alu_ureg`: `set_ureg(32, ..)`, six bits below SM 10.0), and the handle of a
/// bindless constant buffer (`set_src_cb(32..59, ..)`: `v.set_field(0..6, reg)` for
/// `CBuf::BindlessUGPR`).
pub(crate) const SLOT_UNIFORM_REG: Field = Field::new(32, UniformReg::BITS);
/// The constant buffer's byte offset (`set_src_cb`: `v.set_field(6..22, cb.offset)`).
pub(crate) const CBUF_OFFSET: Field = Field::new(38, 16);
/// The bound constant buffer's index (`set_src_cb`: `v.set_field(22..27, idx)`).
pub(crate) const CBUF_INDEX: Field = Field::new(54, 5);
/// The wide slot's uniform bit: for a constant buffer, set when it is bindless (`encode_alu_cb`:
/// `set_src_cb(32..59, 91, ..)`, the "cx" bit); for a uniform register always set
/// (`encode_alu_ureg`: `set_bit(91, true)`).
pub(crate) const SLOT_UNIFORM: Field = Field::bit(91);
/// ALU source 2 as a 32-bit register (`encode_alu_src2`: `set_alu_reg(64..72, 74, 75, ..)` when
/// the source type is not fp16); operand forms 2, 3 and 7 put source 1 here instead.
pub(crate) const SRC2: Field = Field::new(64, Gpr::BITS);
/// ALU source 2's absolute-value bit (32-bit sources).
pub(crate) const SRC2_ABS: Field = Field::bit(74);
/// ALU source 2's negation bit (32-bit sources).
pub(crate) const SRC2_NEG: Field = Field::bit(75);

/// Source 0's swizzle when its type is a packed pair of 16-bit floats (`encode_alu_src0`:
/// `set_alu_reg(24..32, 73, 72, 74..76, ..)`, which calls `set_swizzle(74..76, ..)` for an fp16
/// source; slice 6c, `op::half_double`).
pub(crate) const SRC0_SWIZZLE: Field = Field::new(74, HalfSwizzle::BITS);
/// The wide slot's swizzle for a packed-half source (`encode_alu_reg`, `encode_alu_cb` and
/// `encode_alu_ureg`: `set_swizzle(60..62, ..)` for an fp16 source).
pub(crate) const SLOT_SWIZZLE: Field = Field::new(60, HalfSwizzle::BITS);
/// The register slot's swizzle for a packed-half source (`encode_alu_src2`: `set_alu_reg(64..72,
/// 83, 84, 81..83, ..)` when the source type is fp16).
pub(crate) const SRC2_HALF_SWIZZLE: Field = Field::new(81, HalfSwizzle::BITS);
/// The register slot's absolute-value bit for a packed-half source (`encode_alu_src2`: `if
/// src_type.is_fp16() { 83 } else { 74 }`).
pub(crate) const SRC2_HALF_ABS: Field = Field::bit(83);
/// The register slot's negation bit for a packed-half source (`encode_alu_src2`: `if
/// src_type.is_fp16() { 84 } else { 75 }`).
pub(crate) const SRC2_HALF_NEG: Field = Field::bit(84);

/// The bits every form explains: opcode, guard and scheduling.
pub(crate) const COMMON: u128 = OPCODE.mask() | GUARD.mask() | GUARD_NOT.mask() | SCHEDULING.mask();

/// The bits of `fields`.
pub(crate) const fn mask_of(fields: &[Field]) -> u128 {
    let mut mask = 0;
    let mut i = 0;
    while i < fields.len() {
        mask |= fields[i].mask();
        i += 1;
    }
    mask
}
